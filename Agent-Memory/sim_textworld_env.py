class SimTextWorldEnv:
    def __init__(self):
        self.room = "kitchen"
        self.inventory = []
        self.drawer_open = False
        self.potato_cooked = False
        self.task_success_reward = 10
        # 定义各步骤奖励
        self.step_rewards = {
            "take key": 1,
            "unlock drawer": 2,
            "take knife": 1,
            "take pot": 1,
            "cook potato": 5
        }
            
    def get_task_success_reward(self):
        return self.task_success_reward
            
    def reset(self):
        obs = f"""
当前位置: {self.room}
可去房间: kitchen(厨房), storage(储藏室), living(客厅), dining(餐厅)
背包: {self.inventory}
目标: 去储藏室拿钥匙 → 回厨房开抽屉拿土豆 → 拿刀 → 拿锅 → 煮土豆做土豆泥
"""
        return obs.strip(), {}

    def step(self, action):
        obs = ""
        done = False
        reward = 0  # 初始化奖励
        action = action.strip().lower()

        # 移动
        if action == "go storage":
            self.room = "storage"
            obs = "你来到储藏室，钥匙就在这里！"
        elif action == "go kitchen":
            self.room = "kitchen"
            obs = "你回到厨房"
        elif action == "go living":
            self.room = "living"
            obs = "你来到客厅"
        elif action == "go dining":
            self.room = "dining"
            obs = "你来到餐厅"

        # 拿钥匙：**只要在储藏室，一定能拿到**
        elif action == "take key":
            if self.room == "storage" and "key" not in self.inventory:
                self.inventory.append("key")
                obs = "✅ 拿到钥匙"
                reward = self.step_rewards["take key"]  # 成功拿钥匙奖励
            else:
                obs = "当前位置没有钥匙"

        # 开抽屉
        elif action == "unlock drawer":
            if self.room == "kitchen" and "key" in self.inventory:
                self.inventory.append("potato")
                self.drawer_open = True
                obs = "✅ 打开抽屉，拿到土豆"
                reward = self.step_rewards["unlock drawer"]  # 成功开抽屉奖励
            else:
                obs = "无法打开抽屉"

        # 拿刀
        elif action == "take knife":
            if self.room == "kitchen" and "knife" not in self.inventory:
                self.inventory.append("knife")
                obs = "✅ 拿到菜刀"
                reward = self.step_rewards["take knife"]  # 成功拿刀奖励
            else:
                obs = "无法拿到菜刀"

        # 拿锅
        elif action == "take pot":
            if self.room == "living" and "pot" not in self.inventory:
                self.inventory.append("pot")
                obs = "✅ 拿到煮锅"
                reward = self.step_rewards["take pot"]  # 成功拿锅奖励
            else:
                obs = "无法拿到煮锅"

        # 煮土豆
        elif action == "cook potato":
            if {"potato", "knife", "pot"}.issubset(self.inventory):
                self.potato_cooked = True
                done = True
                obs = "✅ 成功煮好土豆泥！任务完成！"
                reward = self.step_rewards["cook potato"]  # 完成任务奖励
            else:
                obs = "缺少道具，无法煮土豆"

        elif action == "inventory":
            obs = f"背包: {self.inventory} | 当前位置: {self.room}"
        else:
            obs = "无效动作"

        return obs, reward, done, {}