import argparse
#选择模型，batch_size，epochs，学习率，dropout概率，是否使用cuda，隐藏层大小
parser = argparse.ArgumentParser()
parser.add_argument("--model",type=str,default="lstm",choices=["cnngru","gru","lstm","cnnlstm"])
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--drop_prob", type=float, default=0.6)
parser.add_argument("--iscuda",type=bool,default=True)
parser.add_argument("--hidden_size",type=int,default=256)

args = parser.parse_args()

if __name__ == "__main__":
    pass