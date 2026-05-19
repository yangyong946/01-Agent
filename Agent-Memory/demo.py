# demo.py
from knowledge_agent import ReActAgent
from sim_textworld_env import SimTextWorldEnv
from memory_module import MemoryModule
import time
import datetime

def run_demo():
    # 初始化环境、记忆模块和智能体
    env = SimTextWorldEnv()
    memory = MemoryModule()
    agent = ReActAgent(memory)
    obs, _ = env.reset()
    
    # 初始化统计变量
    total_steps = 0
    successful_steps = 0
    task_success = False
    retrieval_times = []  # 存储每次记忆检索的耗时
    memory_usage_success = 0  # 使用记忆后执行成功的次数
    memory_usage_total = 0    # 总共使用记忆的次数
    total_reward = 0  # 新增：总奖励统计
    step_rewards = []  # 新增：每步奖励列表

    print("===== 记忆增强型Agent Demo 开始 =====")
    print(f"初始状态：{obs}")
    done = False
    step = 0

    while not done and step < 40:
        step += 1
        total_steps = step
        time.sleep(1)  # 放慢速度，方便录屏
        
        # 统计记忆检索耗时
        retrieve_start = time.time()
        think, action = agent.act(obs)
        retrieve_end = time.time()
        retrieval_time = retrieve_end - retrieve_start
        retrieval_times.append(retrieval_time)
        
        print(f"\n===== 第 {step} 步 =====")
        print(f"思考：{think}")
        print(f"执行动作：{action}")
        
        # 保存执行前的obs
        prev_obs = obs  
        obs, reward, done, _ = env.step(action)
        
        # 新增：奖励统计
        total_reward += reward
        step_rewards.append(reward)
        
        print(f"执行结果：{obs}")
        print(f"即时奖励：{reward}")  # 现在会显示对应奖励
        
        # 存储记忆
        memory.add_short(f"step{step}: {action}")
        memory.add_long(prev_obs, action, obs)
        
        # 修正：步骤成功判定（动作推进任务/执行成功，而非仅reward>0）
        step_success = False
        if "✅" in obs or "任务完成" in obs:  # 执行结果包含成功标记则步骤成功
            step_success = True
            successful_steps += 1
        
        # 修正：记忆使用统计（思考中包含记忆相关关键词则判定为使用记忆）
        memory_related_keywords = ["记忆", "检索", "已获取道具", "子任务", "前置条件"]
        if any(keyword in think for keyword in memory_related_keywords):
            memory_usage_total += 1
            if step_success:
                memory_usage_success += 1
        
        # 修正：任务成功判定（done=True且执行结果包含任务完成）
        if done and "任务完成" in obs:
            task_success = True

    # 计算统计指标
    # 任务成功率
    task_success_rate = 1.0 if task_success else 0.0
    # 步骤成功率
    step_success_rate = successful_steps / total_steps if total_steps > 0 else 0.0
    # 记忆检索平均速度
    avg_retrieval_speed = sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0.0
    # 记忆使用成功率
    memory_success_rate = memory_usage_success / memory_usage_total if memory_usage_total > 0 else 0.0
    # 新增：平均每步奖励
    avg_step_reward = total_reward / total_steps if total_steps > 0 else 0.0

    # 打印最终统计结果
    print("\n" + "="*50)
    print("===== Demo 执行完毕 - 统计结果 =====")
    print(f"总执行步数：{total_steps}")
    print(f"任务成功率：{task_success_rate:.2%}")
    print(f"步骤成功率：{step_success_rate:.2%}")
    print(f"平均记忆检索耗时：{avg_retrieval_speed:.4f} 秒")
    print(f"记忆使用次数：{memory_usage_total}")
    print(f"记忆使用成功率：{memory_success_rate:.2%}")
    # 新增奖励相关统计
    print(f"总奖励：{total_reward}")
    print(f"平均每步奖励：{avg_step_reward:.2f}")
    print(f"各步奖励列表：{step_rewards}")
    print("="*50)

if __name__ == "__main__":
    run_demo()