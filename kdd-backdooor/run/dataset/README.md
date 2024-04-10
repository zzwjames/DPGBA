# dataset
### Type 1: Hard to answer without LLM
#### Type 1-1 Images with fog/rain/shadow/night (natural global mosaic/The far end of the picture or details are blurry)​ 
1.![image](https://github.com/wzongyu/dataset/assets/131548479/e6259078-5629-484e-a3c1-407fd0d7e881)
这是我们需要的模型出错的例子，事实上，通过车中的“人头”，汽车的正在运行，路边的人影，我们不难看出图片中有人。这种情况就是模型先验知识中缺乏对场景的理解能力以及对“车与人经常共现”的把握。
![image](https://github.com/wzongyu/dataset/assets/131548479/9f3eed05-e9ed-4f74-ac9b-d7b8b11e23e9)
![image](https://github.com/wzongyu/dataset/assets/131548479/ddf02e0c-9ef1-4afc-8659-48f61ea60eef)
![image](https://github.com/wzongyu/dataset/assets/131548479/b5ef6118-e66a-468f-9177-c3567ffa85c9)

类似的例子有：   

2.    
![image](https://github.com/wzongyu/dataset/assets/131548479/7b261b24-7ba6-4dd6-bea1-c1976dcb8578)
  
3.  
![image](https://github.com/wzongyu/dataset/assets/131548479/b6320410-239e-4c4a-adc1-9f33ce808af5)
模型出错，事实上可以通过“鼠标垫，电脑，键盘与鼠标经常共现”!的先验知识，我们很轻易可以“看出来”图片中有鼠标的存在。
![image](https://github.com/wzongyu/dataset/assets/131548479/ef6c11cf-5a69-4ca2-b6c9-99e8c5816c61)
![image](https://github.com/wzongyu/dataset/assets/131548479/e37f6769-a0bc-4ceb-9a13-972c579b2968)
![image](https://github.com/wzongyu/dataset/assets/131548479/b7426b0d-7e2b-4f6c-af14-1be3e1f61795)

### Type 2: Impossible to answer without LLM

#### Type 2-1 Answer rely on the context：
参考[[2402.13607] CODIS: Benchmarking Context-Dependent Visual Comprehension for Multimodal Large Language Models (arxiv.org)](https://arxiv.org/abs/2402.13607)
![image](https://github.com/wzongyu/dataset/assets/131548479/780ff762-17fc-4946-a48a-97ecc6860821)

#### Type 2-2 Some parts are missing or hidden (not shown or snow)​:
![image](https://github.com/wzongyu/dataset/assets/131548479/df6dbf5b-95ed-4db5-937b-d496613f5ad8)
这是模型利用先验知识成功回答问题的例子，尽管图片里没有出现电源，但是通过“手机在充电”这一事实，我们推断出有电源，如若模型回答“no”，就是我们希望的幻觉的例子。
![image](https://github.com/wzongyu/dataset/assets/131548479/b80dedc9-7425-48c6-8747-3a9a5877ad12)
这是模型成功发挥先验知识回答正确我们的问题的例子，即使球在图中不可见，我们仍然可以通过人的姿势推断出球的走向。
![image](https://github.com/wzongyu/dataset/assets/131548479/7706a342-39a0-437e-a880-880bf4c364b7)
  这是我们希望的模型失败的例子。



