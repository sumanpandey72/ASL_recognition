<<<<<<< HEAD
import pandas as pd
import string

df = pd.read_csv("sign_mnist_test.csv")

alpha = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

for i in range (0,25):
    df_new = df.loc[df['label'] == i]
    filename = str(alpha[i])+".csv"
=======
import pandas as pd
import string

df = pd.read_csv("sign_mnist_test.csv")

alpha = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

for i in range (0,25):
    df_new = df.loc[df['label'] == i]
    filename = str(alpha[i])+".csv"
>>>>>>> 5655fa7d12e3a971a4f5d4c17ff968dbedcc5a5a
    df_new.to_csv(filename, index = False)