
from model_regression import train_the_model
import sys

resume = sys.argv[1] if len(sys.argv) > 1 else False

train_the_model(resume)