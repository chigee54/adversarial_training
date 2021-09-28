# Adversarial Training
项目分为两部分：

1、对抗训练自动学习epsilon超参数

2、利用unilm方法训练一个模型，将句子embedding进行还原，输出原始句子。目标是把加扰动的句子embedding还原，查看扰动的效果

3、利用神经网模型将token_embedding还原为句子

Reference:
1. https://github.com/keras-team/keras-tuner/blob/master/examples/cifar10.py
2. https://github.com/bojone/bert4keras/blob/master/examples/task_iflytek_adversarial_training.py
3. https://github.com/bojone/bert4keras/blob/master/examples/task_reading_comprehension_by_seq2seq.py
