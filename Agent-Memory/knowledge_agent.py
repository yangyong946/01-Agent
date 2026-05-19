from config import DASHSCOPE_API_KEY
from memory_module import MemoryModule
from sim_textworld_env import SimTextWorldEnv
# 导入提示词模块
from react_prompts import get_react_agent_prompt
import dashscope
import time  # 计时模块

dashscope.api_key = DASHSCOPE_API_KEY
MAX_STEPS = 40

class ReActAgent:
    def __init__(self, memory):
        self.memory = memory
        self.valid_actions = {
            "go kitchen", "go storage", "go living", "go dining",
            "take key", "unlock drawer", "take knife", "take pot", "cook potato",
            "inventory"
        }
        # 自动阶段：0=拿钥匙 1=开抽屉 2=拿刀 3=拿锅 4=煮土豆
        self.stage = 0

    def act(self, obs):
        # 记录记忆检索开始时间（仅保留计时但不存储，原评估指标相关存储已移除）
        retrieval_start = time.time()
        
        if self.memory:
            # 阶段判断：基于未完成的子任务
            required_items = ["key", "potato", "knife", "pot"]
            obtained = self.memory.obtained_items
            uncompleted = [item for item in required_items if item not in obtained]
            if not uncompleted:
                self.stage = 4  # 所有道具齐，煮土豆
            else:
                self.stage = required_items.index(uncompleted[0])  # 聚焦第一个未完成道具
            
            # 增强错误恢复提示：明确当前房间和未完成道具
            error_correction = f"""
            【关键状态校验】：
            1. 当前所在房间：{self.memory.current_room}
            2. 已获取道具（仅✅确认）：{self.memory.obtained_items}
            3. 未获取的关键道具：{set(['key','potato','knife','pot']) - self.memory.obtained_items}
            4. 动作规则强制校验：
               - take key 必须在 storage 房间
               - unlock drawer 必须在 kitchen + 已拿key
               - take knife 必须在 kitchen
               - take pot 必须在 living
               - cook potato 必须集齐 potato+knife+pot
            """

            memory = self.memory.get_memory_text()
            prompt = get_react_agent_prompt(
                stage=self.stage,
                valid_actions=self.valid_actions,
                obs=obs,
                memory=memory + error_correction
            )
        else:
            # 无记忆模式下直接构建基础prompt
            memory = ""
            prompt = get_react_agent_prompt(
                stage=self.stage,
                valid_actions=self.valid_actions,
                obs=obs,
                memory=memory
            )
        #错误恢复 - 若上一步是失败动作，优先推荐补救动作
        if self.memory and self.memory.short_memory:
            last_step = self.memory.short_memory[-1] if self.memory.short_memory else ""
            last_action = last_step.split(":")[-1].strip() if ":" in last_step else ""
            last_result = [m["result"] for m in self.memory.long_memory if m["action"] == last_action]
            if last_result and "无法" in last_result[-1]:
                # 生成补救提示
                remedy_prompt = f"""
                【错误复盘】上一步执行{last_action}失败，原因：{last_result[-1]}
                请优先选择能修复该错误的动作，例如：
                - 若take key失败→先执行go storage
                - 若unlock drawer失败→先确认是否有key且在kitchen
                - 若take pot失败→先执行go living
                """
                prompt += remedy_prompt

        # 调用大模型生成思考和动作
        resp = dashscope.Generation.call(
            model="qwen-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        try:
            text = resp.output["text"]
            think = text.split("思考：")[-1].split("行动：")[0].strip()
            action = text.split("行动：")[-1].strip().split("\n")[0].strip()
            if action not in self.valid_actions:
                action = "inventory"
        except:
            think = "查看背包"
            action = "inventory"
        return think, action
    
    def _get_current_needed_item(self):
        """获取当前需要的道具（原评估指标相关，保留但无调用）"""
        required_items = ["key", "potato", "knife", "pot"]
        obtained = self.memory.obtained_items if self.memory else set()
        uncompleted = [item for item in required_items if item not in obtained]
        return uncompleted[0] if uncompleted else "cook potato"


def main():
    env = SimTextWorldEnv()
    memory = MemoryModule()
    agent = ReActAgent(memory)
    obs, _ = env.reset()
    done = False
    step = 0
    success = False
    while not done and step < MAX_STEPS:
        step += 1
        print(f"\n===== 第 {step} 步 =====")
        print(f"观察: {obs}")
        print(f"【当前记忆】已获取道具：{memory.obtained_items} | 当前房间：{memory.current_room}")
        think, action = agent.act(obs)
        print(f"思考: {think}")
        print(f"动作: {action}")
        
        # 关键修复：保存执行动作前的obs为prev_obs
        prev_obs = obs  
        obs, reward, done, _ = env.step(action)
        
        print(f"结果: {obs}")
        memory.add_short(f"step{step}: {action}")
        memory.add_long(prev_obs, action, obs)  # 现在prev_obs已定义
    
    # 标记任务是否成功
    success = done
    
    print("\n【最终结果】:", "✅ 任务成功" if done else "❌ 任务失败")

if __name__ == "__main__":
    main()