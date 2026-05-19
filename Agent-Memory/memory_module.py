import time
from collections import defaultdict
from difflib import get_close_matches
import faiss
#from modelscope import snapshot_download
#model_dir = snapshot_download('../all-MiniLM-L6-v2')
# 导入SentenceTransformer（向量模型）
from sentence_transformers import SentenceTransformer
import numpy as np


class MemoryModule:
    def __init__(self):
        self.short_memory = []    # 短期记忆（单轮episode步骤）
        self.long_memory = []     # 长期记忆（跨episode经验）
        self.visited_rooms = set()
        self.obtained_items = set()
        self.completed_steps = set()
        # 新增：RAG 检索库（长期记忆向量化/检索）
        self.rag_index = defaultdict(list)  # key: 场景关键词，value: 经验记录
        self.current_room = ""  # 新增：记录当前房间
        self.embed_model = SentenceTransformer('../all-MiniLM-L6-v2')  # 轻量向量模型
        self.rag_index_vec = faiss.IndexFlatL2(384)  # 向量维度匹配模型输出
        self.vec_to_mem = {}  # 向量索引→记忆数据的映射 
        retrieval_stats = {
            "total_calls": 0,
            "success_calls": 0,
            "total_time": 0.0
        }        
            
    # 补全缺失的add_short方法
    def add_short(self, content):
        self.short_memory.append(content)
        # 从content中提取当前房间（比如step1: go storage → storage）
        if "go " in content:
            self.current_room = content.split("go ")[-1].strip()


    def add_long(self, state, action, result):
        self.long_memory.append({"state": state, "action": action, "result": result})
        if "来到" in result: 
            room = action.split()[-1] if len(action.split())>1 else ""
            self.visited_rooms.add(room)
            self.current_room = room
        if "✅ 拿到" in result: 
            item = result.split("拿到")[-1].replace("✅", "").strip()
            item_map = {"钥匙": "key", "土豆": "potato", "菜刀": "knife", "煮锅": "pot"}
            item = item_map.get(item, item)
            self.obtained_items.add(item)
        # 新增：处理“打开抽屉拿到土豆”的场景
        elif "✅ 打开抽屉，拿到土豆" in result:
            self.obtained_items.add("potato")  # 直接添加potato
        if "✅" in result and "完成" in result: 
            self.completed_steps.add(action)
        
        keywords = [self.current_room, action.split()[0]]
        for kw in keywords:
            if kw:
                self.rag_index[kw].append({"state": state, "action": action, "result": result})
        # 新增：向量化并插入faiss索引
        text = f"{state}→{action}→{result}"
        vec = self.embed_model.encode([text])[0].astype(np.float32)
        idx = self.rag_index_vec.ntotal
        self.rag_index_vec.add(vec.reshape(1, -1))
        self.vec_to_mem[idx] = {"state": state, "action": action, "result": result}
    # 新增：RAG检索函数（根据当前场景找相似经验）
    def retrieve_memory(self, current_obs, top_k=3):
        # 提取当前场景关键词（如房间、背包、目标）
        current_vec = self.embed_model.encode([current_obs])[0].astype(np.float32)
        distances, indices = self.rag_index_vec.search(current_vec.reshape(1, -1), top_k)
        retrieved = []
        for idx in indices[0]:
            if idx != -1 and idx in self.vec_to_mem:
                retrieved.append(self.vec_to_mem[idx])
        return retrieved

    # 新增：获取失败动作列表（被knowledge_agent.py调用）
    def get_bad_actions(self):
        bad_actions = []
        # 新增：过滤重复的成功动作（冗余动作）
        completed_unique = set()
        for mem in self.long_memory:
            if "✅" in mem["result"]:
                if mem["action"] in completed_unique:
                    # 重复的成功动作视为“坏动作”
                    bad_actions.append(mem["action"])
                else:
                    completed_unique.add(mem["action"])
            elif "✅" not in mem["result"] and mem["result"] not in ["无效动作", "背包: [] | 当前位置: kitchen"]:
                bad_actions.append(mem["action"])
        return list(set(bad_actions))

    def get_memory_text(self):
        bad = self.get_bad_actions()
        retrieved = self.retrieve_memory("\n".join(self.short_memory))
        retrieved_text = "\n".join([f"经验：{item['state']} → 动作：{item['action']} → 结果：{item['result']}" for item in retrieved])
        
        # 新增：明确已完成动作列表
        completed_actions = []
        if "key" in self.obtained_items:
            completed_actions.append("take key")
        if "potato" in self.obtained_items:
            completed_actions.append("unlock drawer")
        if "knife" in self.obtained_items:
            completed_actions.append("take knife")
        if "pot" in self.obtained_items:
            completed_actions.append("take pot")

        return f"""
    【禁止重复失败动作】：{','.join(bad) if bad else '无'}
    【已完成动作】：{','.join(completed_actions) if completed_actions else '无'}
    【已探索房间】：{self.visited_rooms if self.visited_rooms else '无'}
    【已获取道具】：{self.obtained_items if self.obtained_items else '无'}
    【已完成任务】：{self.completed_steps if self.completed_steps else '无'}
    【历史相似经验】：{retrieved_text if retrieved_text else '无'}
    """
    
    def extract_general_rules(self):
        """从长期记忆中提炼通用规则（跨episode）"""
        # 1. 统计成功完成任务的动作序列
        success_sequences = []
        for mem in self.long_memory:
            if "✅ 成功煮好土豆泥" in mem["result"]:
                # 回溯该episode的所有动作（从short_memory中匹配）
                episode_steps = [s for s in self.short_memory if f"step{mem['step']}" in s]
                success_sequences.append(episode_steps)
        
        # 2. 提取高频成功路径（通用规则）
        if success_sequences:
            # 简单统计：取出现次数最多的动作序列
            from collections import Counter
            seq_str = ["→".join([s.split(":")[1].strip() for s in seq]) for seq in success_sequences]
            most_common = Counter(seq_str).most_common(1)[0][0]
            general_rules = f"通用成功路径：{most_common}\n"
            
            # 3. 提炼动作前置条件规则（从失败经验）
            bad_actions = self.get_bad_actions()
            bad_rules = []
            for action in bad_actions:
                if "take key" in action and "storage" not in self.current_room:
                    bad_rules.append(f"❌ {action}：不在storage房间执行会失败")
                elif "unlock drawer" in action and "key" not in self.obtained_items:
                    bad_rules.append(f"❌ {action}：无key执行会失败")
            general_rules += "通用失败规则：\n" + "\n".join(bad_rules)
            return general_rules
        return "暂无足够成功经验提炼通用规则"

    def get_retrieval_metrics(self):
        """统计记忆检索的速度和准确率"""
        import time
        start = time.time()
        retrieved = self.retrieve_memory("\n".join(self.short_memory))
        retrieval_time = time.time() - start
        
        # 准确率：检索到的经验是否与当前任务相关（简单判断：包含当前房间/未完成道具）
        current_kw = [self.current_room] + list(set(['key','potato','knife','pot']) - self.obtained_items)
        relevant = 0
        for mem in retrieved:
            if any(kw in mem['state'] or kw in mem['action'] for kw in current_kw):
                relevant += 1
        retrieval_acc = relevant / len(retrieved) if retrieved else 1.0
        
        return {
            "retrieval_time_ms": retrieval_time * 1000,
            "retrieval_accuracy": retrieval_acc,
            "retrieved_count": len(retrieved)
        }
    def retrieve(self, query):
        """记忆检索核心函数"""
        start_time = time.time()  # 计时开始
        retrieval_stats["total_calls"] += 1
        
        # 原有检索逻辑（比如从向量库/文本库找匹配记忆）
        try:
            result = self._actual_retrieve_logic(query)
            # 判定检索成功（比如结果非空/匹配度达标）
            if result is not None and len(result) > 0:
                retrieval_stats["success_calls"] += 1
                success = True
            else:
                success = False
            
            # 计算耗时
            elapsed_time = time.time() - start_time
            retrieval_stats["total_time"] += elapsed_time
            
            # 打印单次检索结果
            print(f"【记忆检索】耗时: {elapsed_time:.4f}s | 本次成功: {success}")
            # 打印累计统计（可选，比如每10次检索打印一次）
            if retrieval_stats["total_calls"] % 10 == 0:
                avg_time = retrieval_stats["total_time"] / retrieval_stats["total_calls"]
                success_rate = retrieval_stats["success_calls"] / retrieval_stats["total_calls"] * 100
                print(f"【记忆检索累计】总次数: {retrieval_stats['total_calls']} | 平均耗时: {avg_time:.4f}s | 成功率: {success_rate:.2f}%")
            
            return result
        except Exception as e:
            # 检索失败也统计耗时
            elapsed_time = time.time() - start_time
            retrieval_stats["total_time"] += elapsed_time
            print(f"【记忆检索】耗时: {elapsed_time:.4f}s | 失败（异常）: {str(e)}")
            raise e        