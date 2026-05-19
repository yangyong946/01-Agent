# experiment.py
from knowledge_agent import ReActAgent
from sim_textworld_env import SimTextWorldEnv
from memory_module import MemoryModule
import numpy as np

def run_agent(use_memory, num_episodes=10):
    """运行多轮实验，统计成功率"""
    success_count = 0
    steps_list = []
    for episode in range(num_episodes):
        env = SimTextWorldEnv()
        memory = MemoryModule() if use_memory else None
        agent = ReActAgent(memory) if use_memory else ReActAgent(None)  # 适配无记忆逻辑
        obs, _ = env.reset()
        done = False
        step = 0
        while not done and step < 40:
            step += 1
            think, action = agent.act(obs)
            prev_obs = obs  # 新增：保存执行前的状态
            obs, reward, done, _ = env.step(action)
            if use_memory:
                agent.memory.add_short(f"step{step}: {action}")
                agent.memory.add_long(prev_obs, action, obs)
        if done:
            success_count += 1
        steps_list.append(step)
    success_rate = success_count / num_episodes
    avg_steps = np.mean(steps_list)
    if use_memory:
        retrieval_metrics = memory.get_retrieval_metrics()
        results["retrieval_time_ms"] = retrieval_metrics["retrieval_time_ms"]
        results["retrieval_accuracy"] = retrieval_metrics["retrieval_accuracy"]    
    return {
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "num_episodes": num_episodes,
        "steps_list": steps_list,  # 新增：返回所有步数
        "success_list": [1 if done else 0 for done in [step <= MAX_STEPS and env.potato_cooked for step in steps_list]]  # 新增：成功标记
    }

if __name__ == "__main__":
    with_memory = run_agent(use_memory=True, num_episodes=50)  # 增加样本量到50
    without_memory = run_agent(use_memory=False, num_episodes=50)
    
    # 统计显著性检验（t检验）
    t_stat, p_value = stats.ttest_ind(with_memory["steps_list"], without_memory["steps_list"])
    
    # 保存结果到表格
    results = pd.DataFrame({
        "模式": ["有记忆", "无记忆"],
        "成功率": [with_memory["success_rate"], without_memory["success_rate"]],
        "平均步数": [with_memory["avg_steps"], without_memory["avg_steps"]],
        "步数标准差": [np.std(with_memory["steps_list"]), np.std(without_memory["steps_list"])]
    })
    results.to_csv("experiment_results.csv", index=False)
    
    # 输出增强
    print("===== 对比实验结果 =====")
    print(f"有记忆模块 - 成功率：{with_memory['success_rate']:.2f}，平均步数：{with_memory['avg_steps']:.1f} (±{np.std(with_memory['steps_list']):.1f})")
    print(f"无记忆模块 - 成功率：{without_memory['success_rate']:.2f}，平均步数：{without_memory['avg_steps']:.1f} (±{np.std(without_memory['steps_list']):.1f})")
    print(f"步数差异显著性：t={t_stat:.2f}, p={p_value:.3f} (p<0.05则差异显著)")
    
    # 提取通用经验
    memory = MemoryModule()
    general_rules = memory.extract_general_rules()
    print("\n===== 跨任务通用经验 =====")
    print(general_rules)