In [208]:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

%matplotlib inline
In [209]:
from pylab import rcParams
rcParams['figure.figsize'] = 20, 10
In [210]:
df=pd.read_csv("Fish.csv")
In [211]:
df.head(10)
Out[211]:
Species	Weight	Length1	Length2	Length3	Height	Width
0	Bream	242.0	23.2	25.4	30.0	11.5200	4.0200
1	Bream	290.0	24.0	26.3	31.2	12.4800	4.3056
2	Bream	340.0	23.9	26.5	31.1	12.3778	4.6961
3	Bream	363.0	26.3	29.0	33.5	12.7300	4.4555
4	Bream	430.0	26.5	29.0	34.0	12.4440	5.1340
5	Bream	450.0	26.8	29.7	34.7	13.6024	4.9274
6	Bream	500.0	26.8	29.7	34.5	14.1795	5.2785
7	Bream	390.0	27.6	30.0	35.0	12.6700	4.6900
8	Bream	450.0	27.6	30.0	35.1	14.0049	4.8438
9	Bream	500.0	28.5	30.7	36.2	14.2266	4.9594
In [212]:
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 159 entries, 0 to 158
Data columns (total 7 columns):
 #   Column   Non-Null Count  Dtype
---  ------   --------------  -----
 0   Species  159 non-null    object
 1   Weight   159 non-null    float64
 2   Length1  159 non-null    float64
 3   Length2  159 non-null    float64
 4   Length3  159 non-null    float64
 5   Height   159 non-null    float64
 6   Width    159 non-null    float64
dtypes: float64(6), object(1)
memory usage: 8.8+ KB
In [213]:
df.describe()
Out[213]:
Weight	Length1	Length2	Length3	Height	Width
count	159.000000	159.000000	159.000000	159.000000	159.000000	159.000000
mean	398.326415	26.247170	28.415723	31.227044	8.970994	4.417486
std	357.978317	9.996441	10.716328	11.610246	4.286208	1.685804
min	0.000000	7.500000	8.400000	8.800000	1.728400	1.047600
25%	120.000000	19.050000	21.000000	23.150000	5.944800	3.385650
50%	273.000000	25.200000	27.300000	29.400000	7.786000	4.248500
75%	650.000000	32.700000	35.500000	39.650000	12.365900	5.584500
max	1650.000000	59.000000	63.400000	68.000000	18.957000	8.142000
In [214]:
corrMatrix = df.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

In [215]:
df.hist()
Out[215]:
array([[<matplotlib.axes._subplots.AxesSubplot object at 0x7f794efb6b50>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x7f794ef36970>],
       [<matplotlib.axes._subplots.AxesSubplot object at 0x7f794eeed250>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x7f794ef15ac0>],
       [<matplotlib.axes._subplots.AxesSubplot object at 0x7f794eecc370>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x7f794ee75bb0>]],
      dtype=object)

In [216]:
del df['Species']
In [217]:
df['Weight']=df['Weight'].astype(float)
In [218]:
df['Length1']=df['Length1'].astype(float)
In [219]:
df['Height']=df['Height'].astype(float)
In [220]:
df['Width']=df['Width'].astype(float)
In [221]:
df['Length2']=df['Length2'].astype(float)
In [222]:
df['Length3']=df['Length3'].astype(float)
In [223]:
def train_test_split(X):
    X = X.sample(frac=1)
    TRAIN_SIZE = int(0.8 * len(X))
    train, test = X.iloc[:TRAIN_SIZE, :], X.iloc[TRAIN_SIZE:, :]

    train_y, test_y = train.pop("Weight"), test.pop("Weight")

    return (train.to_numpy(), test.to_numpy(), train_y.to_numpy(), test_y.to_numpy())

x_train, x_test, y_train, y_test = train_test_split(df)
In [224]:
class MyLinearReg:
    def __init__(self):
        self.coefs = np.random.randn(1, len(x_train[0])) # random vals for each parameter coeff
        self.slope = 0 # slope starts at 0
        self.predictions = []

    def forward_prop(self, X):
        # matrix multiplication between coeffs and X
        # X.shape = n * m   coefs.shape = 1 * n     z.shape = 1 * m
        z = np.dot(self.coefs, X) + self.slope
        self.predictions = z
        # adding the bias AKA slope
        # z --> predicted vals
        return z

    def cost_func(self, z, y):
        # z-> predicted y-> truth
        m = len(y) # m vals (number of vals)
        J = (1/(2 * m)) * np.sum(np.square(z - y)) # getting squared errors
        # multiplying by reciprocal of 2m for cost func
        return J

    def back_prop(self, X, y, z):
        m = len(y) # m vals (number of vals)
        d_coeffs =  (-2 / m) * np.sum((y - z) * X) #gradient of coeffs
        d_slope = (-2 / m) * np.sum((y - z)) #gradient of slopee
        return d_coeffs, d_slope

    def gradient_descent_update(self, d_coeffs, d_slope, lr):
        #updating gradients
        self.coefs = self.coefs - (lr * d_coeffs)
        self.slope = self.slope - (lr * d_slope)

    def normalise_vals(self, arr):
        arr = arr - arr.mean(axis=0)
        arr = arr / np.abs(arr).max(axis=0)
        return arr

    def fit(self, x_train, y_train, x_test, y_test, lr, epochs):
        """
        given X and y fitting a linear reg
        """
        x_train = self.normalise_vals(x_train)
        x_test = self.normalise_vals(x_test)

        cost_train_vals = []
        cost_test_vals = []

        MAE_train_vals = []
        MAE_test_vals = []

        m_train = len(y_train)
        m_val = len(y_test)

        for epoch in range(1, epochs + 1):
            z_train = self.forward_prop(x_train)
            cost_train = self.cost_func(z_train, y_train)
            dw, db = self.back_prop(x_train, y_train, z_train)
            self.gradient_descent_update(dw, db, lr)

            #Mean absolute error MAE
            MAE_train = (1 / m_train) * np.sum(np.abs(z_train - y_train))

            #validation set
            z_val = self.forward_prop(x_test)
            cost_val = self.cost_func(z_val, y_test)
            MAE_val = (1 / m_val) * np.sum(np.abs(z_val - y_test))

            #storing vals for graphing
            cost_train_vals.append(cost_train)
            cost_test_vals.append(cost_val)

            MAE_train_vals.append(MAE_train)
            MAE_test_vals.append(MAE_val)

            #prinitng out
            if epoch % 10 == 0:
                print(f"EPOCHS {epoch} / {epochs} training MAE {MAE_train} validation MAE {MAE_val} training cost {cost_train} validation cost {cost_val}")

        # plotting graph
        fig, ax = plt.subplots(1, 2)

        ax[0].plot(cost_train_vals)
        ax[0].scatter([x for x in range(len(cost_train_vals))], cost_train_vals, label="train_set")
        ax[0].scatter([x for x in range(len(cost_test_vals))], cost_test_vals, label="test_set")
        ax[0].plot(cost_test_vals)
        ax[0].title.set_text('Cost')
        ax[0].legend()

        ax[1].plot(MAE_train_vals)
        ax[1].scatter([x for x in range(len(MAE_train_vals))], MAE_train_vals, label="train_set")
        ax[1].scatter([x for x in range(len(MAE_test_vals))], MAE_test_vals, label="test_set")
        ax[1].plot(MAE_test_vals)
        ax[1].title.set_text('MAE')
        ax[1].legend()


        return (self.slope, self.coefs)

    def R2_val (self, X, y):
        m = len(y)
        z = np.dot(self.coefs, X.T) + self.slope

        # variation across line
        ss = np.sum((y - z) **2)
        ss /= m

        # variation across mean
        mean = np.mean(y)
        var_mean = np.sum((y - mean) ** 2)
        var_mean /= m

        print(f"var across line {ss} || var across mean {var_mean}")
        r2 = 1 - ss / var_mean
        return abs(r2)
In [241]:
linearReg = MyLinearReg()
LEARNING_RATE = 0.01
EPOCHS = 100
slope, coeff = linearReg.fit(x_train.T, y_train, x_test.T, y_test, LEARNING_RATE, EPOCHS)
EPOCHS 10 / 100 training MAE 324.8297104286741 validation MAE 434.75444975983544 training cost 108915.22753212762 validation cost 162068.39532261444
EPOCHS 20 / 100 training MAE 288.2018821188615 validation MAE 399.4816837361724 training cost 92443.81243719775 validation cost 140279.17848719846
EPOCHS 30 / 100 training MAE 272.2827053999161 validation MAE 376.497557615597 training cost 81447.36441373534 validation cost 124840.20813443974
EPOCHS 40 / 100 training MAE 267.0789743836205 validation MAE 361.08958643934545 training cost 74106.04805228132 validation cost 113803.94172500854
EPOCHS 50 / 100 training MAE 264.83766032187305 validation MAE 351.46921605576847 training cost 69204.9267261985 validation cost 105840.32535378679
EPOCHS 60 / 100 training MAE 263.84880763147305 validation MAE 345.0389827417755 training cost 65932.8990583667 validation cost 100037.00746982182
EPOCHS 70 / 100 training MAE 264.73664087065004 validation MAE 339.90087397432546 training cost 63748.467303518984 validation cost 95764.96081122471
EPOCHS 80 / 100 training MAE 266.31608624764596 validation MAE 335.8264412416178 training cost 62290.123250227676 validation cost 92587.95415783557
EPOCHS 90 / 100 training MAE 268.8516984789354 validation MAE 333.86123500705935 training cost 61316.52113468875 validation cost 90201.44790986381
EPOCHS 100 / 100 training MAE 271.01567036548744 validation MAE 332.4598561250966 training cost 60666.53660103734 validation cost 88391.25526738702

In [242]:
df.describe()
Out[242]:
Weight	Length1	Length2	Length3	Height	Width
count	159.000000	159.000000	159.000000	159.000000	159.000000	159.000000
mean	398.326415	26.247170	28.415723	31.227044	8.970994	4.417486
std	357.978317	9.996441	10.716328	11.610246	4.286208	1.685804
min	0.000000	7.500000	8.400000	8.800000	1.728400	1.047600
25%	120.000000	19.050000	21.000000	23.150000	5.944800	3.385650
50%	273.000000	25.200000	27.300000	29.400000	7.786000	4.248500
75%	650.000000	32.700000	35.500000	39.650000	12.365900	5.584500
max	1650.000000	59.000000	63.400000	68.000000	18.957000	8.142000
In [243]:
slope, coeff
Out[243]:
(327.5141734181188,
 array([[ 0.14484627,  0.01833429, -1.04383391,  1.0810151 , -0.91817843]]))
In [244]:
r2 = linearReg.R2_val(x_test, y_test)
print("-"*20)
print(f"R SQUARED = {r2}")
var across line 190985.91567961866 || var across mean 152521.91561523438
--------------------
R SQUARED = 0.25218670975400714
In [245]:
fig, ax = plt.subplots()
ax.scatter([x for x in range(y_test.shape[0])], y_test, label="actual")
ax.scatter([x for x in range(linearReg.predictions.shape[1])], linearReg.predictions, label = "predictions")
ax.legend()
plt.title("Actual  VS predicted")
plt.show()

In [246]:
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(x_train, y_train)
In [247]:
print(f"{model.intercept_}\n{model.coef_}")
-489.0096632239906
[ 67.88346422 -10.7507045  -29.56369354  28.46849806  16.50488414]
In [248]:
print(f"Model score (R squared) val with training set {model.score(x_train, y_train)}")
print("--"*20)
print(f"Model score (R squared) val with testing set {model.score(x_test, y_test)}")
Model score (R squared) val with training set 0.8879590329135785
----------------------------------------
Model score (R squared) val with testing set 0.8674518406443346
In [ ]:

In [101]:

In [102]:
df.describe()
Out[102]:
Species	Weight	Length1	Length2	Length3	Height	Width
count	159	159	159	159	159	159	159
unique	7	101	116	93	124	154	152
top	Perch	300.0	19.0	22.0	23.5	2.2139	3.525
freq	56	6	6	7	5	2	3
In [103]:
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 159 entries, 0 to 158
Data columns (total 7 columns):
 #   Column   Non-Null Count  Dtype
---  ------   --------------  -----
 0   Species  159 non-null    object
 1   Weight   159 non-null    object
 2   Length1  159 non-null    object
 3   Length2  159 non-null    object
 4   Length3  159 non-null    object
 5   Height   159 non-null    object
 6   Width    159 non-null    object
dtypes: object(7)
memory usage: 8.8+ KB
In [ ]:

In [ ]:

In [ ]:

In [ ]:

In [ ]:

In [ ]:

In [ ]:

In [ ]:

In [ ]:

In [ ]:

In [ ]:
