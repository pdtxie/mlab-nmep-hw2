import pandas as pd
import numpy as np

preds = np.load("./output/resnet18/preds.npy")

final_preds = np.argmax(preds, axis=1)

df = pd.DataFrame(data = final_preds,  
                  columns = ["Category"]) 

df.to_csv("test.csv")
