Before proceeding to the most interesting tasks, we should take care of
the data, which are quite dirty

Let's import the required libraries, and then proceed to the first
dataset

In [1]:

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

In [2]:

    Demographics = pd.read_csv('Model Build - Demographics.csv', header=0, encoding='ISO-8859-15')

In [3]:

    Demographics.head()

Out[3]:

Client ID

Age

Gender 1: Female, 2: Male

County

Income Group

Unnamed: 5

0

1

36

1

Cork

10001 - 40000

NaN

1

2

43

1

Cavan

0 - 10000

NaN

2

3

32

0

Dublin

10001 - 40000

NaN

3

4

52

1

Louth

40001 - 60000

NaN

4

5

63

0

Kilkenny

60001 - 100000

NaN

There is an extra column (extra comma in the csv)

In [4]:

    Demographics = Demographics[Demographics.columns[:5]]

Don't like that (wrong) "Gender 1: Female, 2: Male"

In [5]:

    Demographics = Demographics.rename(columns={'Gender \n1: Female, 2: Male' : "Gender"})
    Demographics.head()

Out[5]:

Client ID

Age

Gender

County

Income Group

0

1

36

1

Cork

10001 - 40000

1

2

43

1

Cavan

0 - 10000

2

3

32

0

Dublin

10001 - 40000

3

4

52

1

Louth

40001 - 60000

4

5

63

0

Kilkenny

60001 - 100000

let's see if there are more entries for the same Client ID

In [6]:

    Demographics.groupby("Client ID").filter(lambda x: len(x) > 1)

Out[6]:

Client ID

Age

Gender

County

Income Group

45

46

56

0

Kerry

10001 - 40000

46

46

20

0

Laois

10001 - 40000

1220

1220

67

0

Tipperary

10001 - 40000

1221

1220

48

1

Kildare

100000+

3675

3674

46

0

Galway

10001 - 40000

3676

3674

37

1

Carlow

10001 - 40000

3677

3675

56

1

Dublin

0 - 10000

3678

3675

23

1

Cork

10001 - 40000

Ok now we should think about these records. They could be meant as
mistakes in the first immissions, with a failed attempt to overwrite.
However, the same exacts tuples are found duplicated in the other
datasets, so their data could easily be real. We do not to delete this
customers from the db, let's assign them a new Client ID.

In [7]:

    Demographics.ix[46, "Client ID"] =  10001
    Demographics.ix[1221, "Client ID"] =  10002
    Demographics.ix[3676, "Client ID"] =  10003
    Demographics.ix[3678, "Client ID"] =  10004
    Demographics.groupby("Client ID").filter(lambda x: len(x) > 1)

Out[7]:

Client ID

Age

Gender

County

Income Group

Focus on Gender

In [8]:

    Demographics.Gender.unique()

Out[8]:

    array([u'1', u'0', u'M', u'Male', u'f', u'fem', u'm', u'female'], dtype=object)

a lot of values (the funny thing is that none of them is 2 as in the
column description) Let's fix this

In [9]:

    def gender (x):
        """Uniform the gender feature"""
        if ((x == '0') or (x == 'f') or (x == 'female') or (x == 'fem')):
            return 0
        else:
            return 1
        
    Demographics['Gender'] = map (lambda x: gender(x), Demographics['Gender'])
    Demographics.Gender.unique()

Out[9]:

    array([1, 0], dtype=int64)

In [10]:

    Demographics.describe()

Out[10]:

Client ID

Age

Gender

count

10004.00000

10004.000000

10004.000000

mean

5002.50000

44.392043

0.509096

std

2888.05038

15.202030

0.499942

min

1.00000

20.000000

0.000000

25%

2501.75000

32.000000

0.000000

50%

5002.50000

44.000000

1.000000

75%

7503.25000

57.000000

1.000000

max

10004.00000

200.000000

1.000000

Ok fine with these. Have a look to the "County field"

In [11]:

    Demographics.County.describe()

Out[11]:

    count      10001
    unique        66
    top       Dublin
    freq        4655
    Name: County, dtype: object

In [12]:

    Demographics.County.value_counts()[0:10]

Out[12]:

    Dublin      4655
    Cork        1336
    Offaly       354
    Clare        295
    Louth        264
    Kildare      197
    Laois        196
    Longford     196
    Kerry        193
    Cavan        192
    Name: County, dtype: int64

In [13]:

    Demographics.County.value_counts()[0:10].plot(kind="bar")
    plt.title("Top 10 Counties")
    plt.ylabel("number of customers")

    plt.show()

![](data:image/png;base64,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%0Ajxo3bpw++ugjZ8cqkLt9yDz11FNatGiRWrdurfbt26tp06bOjmRTqVKlJEkXL15UqVKl3GLf9rVn%0ALZw6dUoVKlRwcqKCOXuzZnFERUXphx9+UGpqqnr27KkOHTo4O5JN2dnZysnJkYeHh1scp3Hvvfdq%0AwYIFOnXqlDp27Ki7777b2ZFsunLlivr376+aNWsqLCxMLVu2dOjzc5BdPp555hl9/PHH1ttPP/20%0AFi1a5LxANixbtkw9e/bUpk2b9Ntvv6lcuXIuu+Yza9YsjRw5UmvXrrXu90tPT3f5eaWlq5NqnD17%0AVj4+Plq7dq3KlCmT59+JK8qdCEm6Okvc/fffL09PTycmKti+ffu0b98+3X777ZoyZYq6du2qZ555%0AxtmxCuSOB4x++OGHWrNmjRo2bKidO3eqc+fOLv0+Dx8+XO3atdOKFSs0evRoxcTEOPSAtVuxc+dO%0ALVy4UCkpKVqzZo3Dnpc1+Ovk7q8sXbq0FixYoObNm2vnzp0ufaGOuXPnWmeE69Chg+rUqaPp06fr%0A/PnzGjJkiLPj3WD16tWqXLmyYmNjdfr06Ty/c+XTdCTlmTGrffv2qlGjhvPCFNK9996rd955R6mp%0AqapRo4aqV6+ucuXKOTtWvurWrau6detKunGeBFflTgeMrl69Wo8++qhCQkLUpk0bHThwQL169VJw%0AcLCzoxXo3Llz6tWrl7788ks1adJEOTk5zo5k06VLl7RmzRp9/vnnMgxDw4YNc+jzU/DXyZ2woly5%0Acjpw4IB1f6UrT6iwfv16JSQkWDexBQUFadasWerTp49LFvxbb72lDRs26PLlyy4//3WuazdzX8/V%0AD/QZN26cmjdvrq5du2r79u0aM2aM5s+f7+xYN8idsyErK0sXL17UHXfcoZMnTyowMFDff/+9k9MV%0AzJ0OGH377bdVp04djR8/XjNmzLBu6j548KBLX6NAujpHgnR1OmNX3gqVq2vXrgoJCdErr7yi6tWr%0AO/z5KfjruPp0tDdTpkyZG8rH29tbfn5+TkpUsPj4eE2bNk2GYbj0ebfX6t27tw4ePKhq1arJ29tb%0AP/74owIDA1WrVi1nR7Pp7Nmz1nnp69Wr59BNhEWRu/Vs9OjRGjVqlLXg3eFv0p0OGA0PD9cbb7yh%0AgwcPauLEidZxV71GwYkTJ1SlShVNmDBB48aNU2pqqoYPH+7SszHm+uabb6wHPktXj4GpXLmyw56f%0A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%0A3dMAAAAASUVORK5CYII=%0A)

In [14]:

    Demographics[Demographics["County"].isnull()]

Out[14]:

Client ID

Age

Gender

County

Income Group

9799

9796

53

1

NaN

60001 - 100000

9895

9892

63

0

NaN

60001 - 100000

9971

9968

43

0

NaN

40001 - 60000

Almost half of the customers live in Dublin. They are just 3...Let's
assume they belong to Dublin as well.

In [15]:

    Demographics = Demographics.fillna(Demographics['County'].value_counts().index[0])

Focus now on Income Group

In [16]:

    Demographics['Income Group'].unique()

Out[16]:

    array([u'10001 - 40000', u'0 - 10000', u'40001 - 60000', u'60001 - 100000',
           u'100000+', u'0  ? 10000', u'10002 - 40000', u'10001 ?. 40000',
           u'10001 ? 40000'], dtype=object)

fix the wrong values

In [17]:

    Demographics.ix[Demographics['Income Group'] == "10001 ?. 40000", "Income Group"] = "10001 - 40000"
    Demographics.ix[Demographics['Income Group'] == "10001 ? 40000", "Income Group"] = "10001 - 40000"
    Demographics.ix[Demographics['Income Group'] == "10001 ? 40000", "Income Group"] = "10001 - 40000"
    Demographics.ix[Demographics['Income Group'] == "10002 - 40000", "Income Group"] = "10001 - 40000"
    Demographics.ix[Demographics['Income Group'] == "0  ? 10000", "Income Group"] = "0 - 10000"

In [18]:

    Demographics['Income Group'].unique()

Out[18]:

    array([u'10001 - 40000', u'0 - 10000', u'40001 - 60000', u'60001 - 100000',
           u'100000+'], dtype=object)

In a joint dataset obtained from Demographics and Counties information,
I will realize that some of the counties names are misspelled. I correct
it now.

In [19]:

    Demographics.ix[Demographics['County'].str.contains("Dub"),"County"] = "Dublin"
    Demographics.ix[Demographics['County'].str.contains("dub"),"County"] = "Dublin"
    Demographics.ix[Demographics['County'].str.contains("blin"),"County"] = "Dublin"
    Demographics.ix[Demographics['County'].str.contains("cork"),"County"] = "Cork"
    Demographics.ix[Demographics['County'].str.contains("Cork"),"County"] = "Cork"
    Demographics.ix[Demographics['County'].str.contains("ongford"),"County"] = "Longford"

Ok let's move to the Lookup of the Counties

In [20]:

    County = pd.read_excel("County Lkp.xlsx", header=[0])
    County.head()

Out[20]:

SNo

County

Population

Density (/ km2)

Rank

Province

Change since

0

NaN

NaN

NaN

NaN

NaN

NaN

previous census

1

1

Dublin

1273069.0

1380.8

1

Leinster

0.0702

2

2

Antrim

618108.0

202.9

2

Ulster

0.018

3

3

Down

531665.0

215.6

3

Ulster

0.0872

4

4

Cork

519032.0

69.0

4

Munster

0.0784

In [21]:

    County.tail(10)

Out[21]:

SNo

County

Population

Density (/ km2)

Rank

Province

Change since

34

29

Monaghan

60483.0

46.7

29

Ulster

0.0801

35

30

Carlow

54612.0

60.8

30

Leinster

0.0847

36

31

Longford

39000.0

35.7

31

Leinster

0.134

37

32

Leitrim

31796.0

19.9

32

Connacht

0.0984

38

NaN

Average

199974.0

NaN

NaN

NaN

NaN

39

NaN

Ireland

6469688.0

76.6

Total

NaN

NaN

40

NaN

NaN

NaN

NaN

NaN

NaN

NaN

41

For other more formats kindly visit www.downlo...

NaN

NaN

NaN

NaN

NaN

NaN

42

NaN

NaN

NaN

NaN

NaN

NaN

NaN

43

Original source : en.wikipedia.org/wiki/List\_o...

NaN

NaN

NaN

NaN

NaN

NaN

Even this file is a bit messy.

In [22]:

    County = County.rename(columns={'Change since' : "Change"})
    County = County[1:-6]

for the rank we can use the population, for the SNo it is probably
better no to reorder what we have already

In [23]:

    County = County.sort_values(["Population"], ascending=False)
    County.head()

Out[23]:

SNo

County

Population

Density (/ km2)

Rank

Province

Change

1

1

Dublin

1273069.0

1380.8

1

Leinster

0.0702

2

2

Antrim

618108.0

202.9

2

Ulster

0.018

3

3

Down

531665.0

215.6

3

Ulster

0.0872

4

4

Cork

519032.0

69.0

4

Munster

0.0784

5

-

Fingal

273991.0

600.6

-

Leinster

0.1417

In [24]:

    County.Rank = np.arange(1,len(County)+1)
    County.head()

Out[24]:

SNo

County

Population

Density (/ km2)

Rank

Province

Change

1

1

Dublin

1273069.0

1380.8

1

Leinster

0.0702

2

2

Antrim

618108.0

202.9

2

Ulster

0.018

3

3

Down

531665.0

215.6

3

Ulster

0.0872

4

4

Cork

519032.0

69.0

4

Munster

0.0784

5

-

Fingal

273991.0

600.6

5

Leinster

0.1417

In [25]:

    County = County.sort_values(["SNo"])
    County.tail(6)

Out[25]:

SNo

County

Population

Density (/ km2)

Rank

Province

Change

37

32

Leitrim

31796.0

19.9

37

Connacht

0.0984

10

-

Dun Laoghaire-Rathdown

206261.0

1620.1

10

Leinster

0.063

25

-

South Tipperary

88432.0

39.2

25

Munster

0.0626

5

-

Fingal

273991.0

600.6

5

Leinster

0.1417

30

-

North Tipperary

70322.0

34.4

30

Munster

0.0661

6

-

South Dublin

265205.0

1190.6

6

Leinster

0.074

In [26]:

    County.iloc[32:,0] = np.arange(33,38)

In [27]:

    County.tail(6)

Out[27]:

SNo

County

Population

Density (/ km2)

Rank

Province

Change

37

32

Leitrim

31796.0

19.9

37

Connacht

0.0984

10

33

Dun Laoghaire-Rathdown

206261.0

1620.1

10

Leinster

0.063

25

34

South Tipperary

88432.0

39.2

25

Munster

0.0626

5

35

Fingal

273991.0

600.6

5

Leinster

0.1417

30

36

North Tipperary

70322.0

34.4

30

Munster

0.0661

6

37

South Dublin

265205.0

1190.6

6

Leinster

0.074

In [28]:

    County.describe()

Out[28]:

SNo

Population

Density (/ km2)

Rank

count

37.000000

3.700000e+01

37.000000

37.000000

mean

19.000000

1.973838e+05

187.621622

19.000000

std

10.824355

2.261722e+05

380.938457

10.824355

min

1.000000

3.179600e+04

19.900000

1.000000

25%

10.000000

7.668700e+04

36.100000

10.000000

50%

19.000000

1.366400e+05

46.800000

19.000000

75%

28.000000

2.062610e+05

119.100000

28.000000

max

37.000000

1.273069e+06

1620.100000

37.000000

Change is not meant as a number

In [29]:

    County["Change"] = map (lambda x: float(x), County["Change"])

Let's go ahead

In [30]:

    PreviousLoan = pd.read_csv('Model Build - Previous Loan Holdings.csv', header=0)
    PreviousLoan.head()

Out[30]:

Client ID

Held Loan previously

0

1

1

1

2

0

2

3

0

3

4

1

4

5

0

In [31]:

    PreviousLoan.groupby("Client ID").filter(lambda x: len(x) > 1)

Out[31]:

Client ID

Held Loan previously

20

21

1

21

21

0

46

46

0

47

46

0

1221

1220

0

1222

1220

0

3676

3674

0

3677

3675

0

3678

3674

0

3679

3675

1

As before

In [32]:

    PreviousLoan.ix[47, "Client ID"] =  10001
    PreviousLoan.ix[1222, "Client ID"] =  10002
    PreviousLoan.ix[3678, "Client ID"] =  10003
    PreviousLoan.ix[3679, "Client ID"] =  10004
    PreviousLoan.drop([21], inplace=True) #we consider the second entry as a correction of the first because this
                                            # specific entry is just here
    PreviousLoan.groupby("Client ID").filter(lambda x: len(x) > 1)

Out[32]:

Client ID

Held Loan previously

In [33]:

    PreviousLoan["Held Loan previously"].describe()

Out[33]:

    count     10004
    unique        4
    top           0
    freq       7500
    Name: Held Loan previously, dtype: object

In [34]:

    PreviousLoan["Held Loan previously"].unique()

Out[34]:

    array(['1', '0', '\xb3', ' \x80-   '], dtype=object)

The not recognized values (depends on the pc) are the euro symbol and a
kind of 3. Probably they should be related to a yes. All values which
are not specified as 0 will be meant as 1.

In [35]:

    PreviousLoan["Held Loan previously"] = map ( lambda x: x.decode("ascii", "ignore"), PreviousLoan["Held Loan previously"])
    PreviousLoan["Held Loan previously"].unique()

Out[35]:

    array([u'1', u'0', u'', u' -   '], dtype=object)

In [36]:

    def loanExc (x):
        try:
            if int(x) == 0:
                return 0
            else:
                return 1
        except ValueError:
            return 1

    PreviousLoan["Held Loan previously"] = map(lambda x: loanExc(x), PreviousLoan["Held Loan previously"])
    PreviousLoan["Held Loan previously"].unique()
    PreviousLoan["Held Loan previously"].describe()

Out[36]:

    count    10004.000000
    mean         0.250300
    std          0.433207
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          1.000000
    max          1.000000
    Name: Held Loan previously, dtype: float64

ok, we have converted the strange values into 1

In [37]:

    BankProducts = pd.read_csv('Model Build - Product Held in Bank.csv', header=None, names=["Client ID","ProductsAmount"])
    BankProducts.head()

Out[37]:

Client ID

ProductsAmount

0

1

4

1

2

4

2

3

2

3

4

2

4

5

1

In [38]:

    BankProducts.describe()

Out[38]:

Client ID

count

10004.000000

mean

4999.361755

std

2887.051613

min

1.000000

25%

2499.750000

50%

4998.500000

75%

7499.250000

max

10000.000000

As before, duplicates IDs.

In [39]:

    BankProducts.groupby("Client ID").filter(lambda x: len(x) > 1)

Out[39]:

Client ID

ProductsAmount

45

46

3

46

46

2

1220

1220

4

1221

1220

4

3675

3674

1

3676

3675

1

3677

3674

2

3678

3675

5

In [40]:

    BankProducts.ix[46, "Client ID"] =  10001
    BankProducts.ix[1221, "Client ID"] =  10002
    BankProducts.ix[3677, "Client ID"] =  10003
    BankProducts.ix[3678, "Client ID"] =  10004
    BankProducts.groupby("Client ID").filter(lambda x: len(x) > 1)

Out[40]:

Client ID

ProductsAmount

In [41]:

    BankProducts["ProductsAmount"].unique()

Out[41]:

    array(['4', '2', '1', '5', '3', '-1', '-2', ' '], dtype=object)

In [42]:

    BankProducts.ProductsAmount.value_counts()

Out[42]:

    3     2019
    2     2012
    4     2006
    5     1992
    1     1969
    -1       4
    -2       1
             1
    Name: ProductsAmount, dtype: int64

Since we will probably feed our models with this feature as numerical
and not categorical, as it is meant as the number of products used, it
would have a sense to replace the " " with the average. The -1 and -2
maybe could be intended as typos, so we delete the minus.

In [43]:

    BankProducts.ix[BankProducts["ProductsAmount"]== " ", "ProductsAmount"] = '3'
    BankProducts.ix[BankProducts["ProductsAmount"]== "-1", "ProductsAmount"] = '1'
    BankProducts.ix[BankProducts["ProductsAmount"]== "-2", "ProductsAmount"] = '2'
    BankProducts["ProductsAmount"] = map (lambda x: int(x), BankProducts["ProductsAmount"])

In [44]:

    BankProducts["ProductsAmount"].unique()

Out[44]:

    array([4, 2, 1, 5, 3], dtype=int64)

In [45]:

    BankProducts.describe()

Out[45]:

Client ID

ProductsAmount

count

10004.00000

10004.000000

mean

5002.50000

3.003099

std

2888.05038

1.409714

min

1.00000

1.000000

25%

2501.75000

2.000000

50%

5002.50000

3.000000

75%

7503.25000

4.000000

max

10004.00000

5.000000

Move on. For this dataset we have realized that the training and the
testing set have been inverted.

In [46]:

    TransactionsOut = pd.read_csv('TEST - Transactions out of Current Account.csv', header=0)
    TransactionsOut.head()

Out[46]:

Client

Num Transactions

Last TXN Amount

Merchant Code

Last Transaction Narrative

Unnamed: 5

Unnamed: 6

Unnamed: 7

0

0

1.0

0.0

NaN

NaN

NaN

NaN

NaN

NaN

NaN

1

2.0

17.0

83.66

7211.0

THE BRIDGE LAUNDRY WICKLOW TOWN

NaN

NaN

NaN

NaN

2

3.0

25.0

526.18

3667.0

LUXOR HOTEL/CASINO LAS VEGAS NV

NaN

NaN

NaN

NaN

3

4.0

13.0

70.68

5712.0

HARVEY NORMAN CARRICKMINES

NaN

NaN

NaN

NaN

4

5.0

39.0

259.07

5999.0

PAYPAL \*PETEWOODWAR 35314369001

NaN

NaN

NaN

NaN

In [47]:

    TransactionsOut.columns

Out[47]:

    Index([u'Client', u'Num Transactions', u'Last TXN Amount', u'Merchant Code',
           u'Last Transaction Narrative', u'Unnamed: 5', u'Unnamed: 6',
           u'Unnamed: 7', u'0'],
          dtype='object')

In [48]:

    TransactionsOut = TransactionsOut[TransactionsOut.columns[:5]]

In [49]:

    TransactionsOut.groupby("Client").filter(lambda x: len(x) > 1)

Out[49]:

Client

Num Transactions

Last TXN Amount

Merchant Code

Last Transaction Narrative

45

46.0

11.0

476.30

7829.0

OMNIPLEX CINEMA LIMERICK LIMERICK CO L

46

46.0

0.0

NaN

NaN

NaN

1220

1220.0

14.0

316.90

3008.0

LUFTHANSA AG2201000000000FRANKFURT

1221

1220.0

14.0

492.88

7311.0

THE IRISH TIMES LTD DUBLIN

3675

3674.0

19.0

838.98

5211.0

B & Q IRELAND TALLAGHT

3676

3674.0

0.0

NaN

NaN

NaN

3677

3675.0

11.0

789.95

3364.0

TICKET DESK CHARLEROI GOSSELIES

3678

3675.0

17.0

509.55

8049.0

FOOT STOP LIMITED NAAS

In [50]:

    TransactionsOut.ix[46, "Client"] =  10001
    TransactionsOut.ix[1221, "Client"] =  10002
    TransactionsOut.ix[3676, "Client"] =  10003
    TransactionsOut.ix[3678, "Client"] =  10004
    TransactionsOut.groupby("Client").filter(lambda x: len(x) > 1)

Out[50]:

Client

Num Transactions

Last TXN Amount

Merchant Code

Last Transaction Narrative

In [51]:

    TransactionsOut.describe()

Out[51]:

Client

Num Transactions

Last TXN Amount

Merchant Code

count

10004.00000

10004.000000

6724.000000

6724.000000

mean

5002.50000

18.288685

451.721841

4817.466984

std

2888.05038

22.959727

258.305702

1820.237256

min

1.00000

0.000000

5.390000

742.000000

25%

2501.75000

0.000000

225.437500

3508.000000

50%

5002.50000

10.000000

450.285000

3793.000000

75%

7503.25000

29.250000

674.852500

5950.000000

max

10004.00000

100.000000

899.860000

9950.000000

In [52]:

    TransactionsOut.head()

Out[52]:

Client

Num Transactions

Last TXN Amount

Merchant Code

Last Transaction Narrative

0

1.0

0.0

NaN

NaN

NaN

1

2.0

17.0

83.66

7211.0

THE BRIDGE LAUNDRY WICKLOW TOWN

2

3.0

25.0

526.18

3667.0

LUXOR HOTEL/CASINO LAS VEGAS NV

3

4.0

13.0

70.68

5712.0

HARVEY NORMAN CARRICKMINES

4

5.0

39.0

259.07

5999.0

PAYPAL \*PETEWOODWAR 35314369001

The NaNs have a sense here. Let's keep them for now.

In [53]:

    TxnAmount = pd.read_csv('Model Build - TXN Amount.csv', header=None, encoding='ISO-8859-15', names = ["Client", "TxnAmount"])
    TxnAmount.head()

Out[53]:

Client

TxnAmount

0

1

€58

1

2

€2,663

2

3

€46

3

4

€0

4

5

€126

In [54]:

    TxnAmount.groupby("Client").filter(lambda x: len(x) > 1)

Out[54]:

Client

TxnAmount

45

46

€0

46

46

€0

1220

1220

€57

1221

1220

€57

3675

3674

€0

3676

3675

€0

3677

3674

€0

3678

3675

€1,636

In [55]:

    TxnAmount.ix[46, "Client"] =  10001
    TxnAmount.ix[1221, "Client"] =  10002
    TxnAmount.ix[3677, "Client"] =  10003
    TxnAmount.ix[3678, "Client"] =  10004
    TxnAmount.groupby("Client").filter(lambda x: len(x) > 1)

Out[55]:

Client

TxnAmount

The euro symbol could give some problems with some configurations. Also
the comma used to count the thousands.

In [56]:

    TxnAmount["TxnAmount"] = map ( lambda x: x.encode("ascii", "ignore"), TxnAmount["TxnAmount"]) #depends on the pc
    TxnAmount["TxnAmount"] = map (lambda x: x.replace(",",""), TxnAmount["TxnAmount"])
    TxnAmount["TxnAmount"] = map (lambda x: float(x), TxnAmount["TxnAmount"])
    TxnAmount.head()

Out[56]:

Client

TxnAmount

0

1

58.0

1

2

2663.0

2

3

46.0

3

4

0.0

4

5

126.0

In [57]:

    Target = pd.read_csv('Target Variable - Purchased Loan Flag.csv', header=0)
    Target.head()

Out[57]:

Client ID

Loan Flag

Unnamed: 2

Unnamed: 3

0

1

0

NaN

NaN

1

2

0

NaN

NaN

2

3

0

NaN

NaN

3

4

0

NaN

NaN

4

5

0

NaN

NaN

In [58]:

    Target = Target[Target.columns[:2]]
    Target.describe()

Out[58]:

Client ID

Loan Flag

count

10004.000000

10004.000000

mean

4999.361755

0.021092

std

2887.051613

0.143697

min

1.000000

0.000000

25%

2499.750000

0.000000

50%

4998.500000

0.000000

75%

7499.250000

0.000000

max

10000.000000

1.000000

In [59]:

    Target.groupby("Client ID").filter(lambda x: len(x) > 1)

Out[59]:

Client ID

Loan Flag

45

46

0

46

46

0

1220

1220

0

1221

1220

0

3675

3674

0

3676

3675

0

3677

3674

0

3678

3675

0

In [60]:

    Target.ix[46, "Client ID"] =  10001
    Target.ix[1221, "Client ID"] =  10002
    Target.ix[3677, "Client ID"] =  10003
    Target.ix[3678, "Client ID"] =  10004
    Target.groupby("Client ID").filter(lambda x: len(x) > 1)

Out[60]:

Client ID

Loan Flag

Ok. let's try to join these datasets to fullfil the BI assignments. I
know that JoinTest is not the best name for this :)

In [61]:

    JoinTest = pd.merge(Demographics,County, how = 'outer', left_on='County', right_on='County')
    JoinTest = pd.merge(JoinTest,PreviousLoan, how = 'outer', left_on='Client ID', right_on='Client ID')
    JoinTest = pd.merge(JoinTest,BankProducts, how = 'outer', left_on='Client ID', right_on='Client ID')
    JoinTest = pd.merge(JoinTest,TransactionsOut, how = 'outer', left_on='Client ID', right_on='Client')
    JoinTest = pd.merge(JoinTest,TxnAmount, how = 'outer', left_on='Client ID', right_on='Client')
    JoinTest = pd.merge(JoinTest,Target, how = 'outer', left_on='Client ID', right_on='Client ID')
    JoinTest.head()

Out[61]:

Client ID

Age

Gender

County

Income Group

SNo

Population

Density (/ km2)

Rank

Province

...

Held Loan previously

ProductsAmount

Client\_x

Num Transactions

Last TXN Amount

Merchant Code

Last Transaction Narrative

Client\_y

TxnAmount

Loan Flag

0

1.0

36.0

1.0

Cork

10001 - 40000

4.0

519032.0

69.0

4.0

Munster

...

1.0

4.0

1.0

0.0

NaN

NaN

NaN

1.0

58.0

0.0

1

9.0

24.0

0.0

Cork

60001 - 100000

4.0

519032.0

69.0

4.0

Munster

...

0.0

4.0

9.0

5.0

45.00

5972.0

DUBLIN MINT OFFICE LONDON

9.0

22.0

0.0

2

24.0

57.0

1.0

Cork

10001 - 40000

4.0

519032.0

69.0

4.0

Munster

...

0.0

2.0

24.0

11.0

890.00

5192.0

EASONS KILKENNY CO DUBLIN

24.0

0.0

0.0

3

37.0

21.0

1.0

Cork

10001 - 40000

4.0

519032.0

69.0

4.0

Munster

...

1.0

1.0

37.0

4.0

247.65

4468.0

MARINE SUPPLIERS DUBLIN

37.0

28.0

0.0

4

38.0

60.0

0.0

Cork

100000+

4.0

519032.0

69.0

4.0

Munster

...

0.0

4.0

38.0

9.0

220.27

3420.0

ENTERPRISE GUERIN RENT FARO

38.0

0.0

0.0

5 rows × 21 columns

In [62]:

    JoinTest['Client ID'].describe()

Out[62]:

    count    10004.00000
    mean      5002.50000
    std       2888.05038
    min          1.00000
    25%       2501.75000
    50%       5002.50000
    75%       7503.25000
    max      10004.00000
    Name: Client ID, dtype: float64

In [63]:

    JoinTest[pd.isnull(JoinTest['SNo'])]

Out[63]:

Client ID

Age

Gender

County

Income Group

SNo

Population

Density (/ km2)

Rank

Province

...

Held Loan previously

ProductsAmount

Client\_x

Num Transactions

Last TXN Amount

Merchant Code

Last Transaction Narrative

Client\_y

TxnAmount

Loan Flag

9980

267.0

65.0

1.0

Navan

10001 - 40000

NaN

NaN

NaN

NaN

NaN

...

0.0

2.0

267.0

0.0

NaN

NaN

NaN

267.0

864.0

0.0

9981

323.0

37.0

1.0

Borris

10001 - 40000

NaN

NaN

NaN

NaN

NaN

...

0.0

2.0

323.0

17.0

740.69

9950.0

Hurst Heating & Plumbin Caslebar, co

323.0

1418.0

0.0

9982

381.0

66.0

1.0

Sandyford

10001 - 40000

NaN

NaN

NaN

NaN

NaN

...

0.0

5.0

381.0

1.0

790.51

5331.0

PAYPAL \*NETMEDIALLC 35314369001

381.0

0.0

0.0

9983

524.0

43.0

0.0

Maynooth

10001 - 40000

NaN

NaN

NaN

NaN

NaN

...

1.0

5.0

524.0

95.0

616.98

5733.0

SOUNDTRAX CAMDEN QUAY

524.0

93.0

1.0

9984

569.0

49.0

1.0

Portlaoise

40001 - 60000

NaN

NaN

NaN

NaN

NaN

...

0.0

3.0

569.0

0.0

NaN

NaN

NaN

569.0

377.0

0.0

9985

654.0

37.0

1.0

Trim

100000+

NaN

NaN

NaN

NaN

NaN

...

0.0

5.0

654.0

0.0

NaN

NaN

NaN

654.0

331.0

0.0

9986

721.0

59.0

1.0

Ballina

60001 - 100000

NaN

NaN

NaN

NaN

NaN

...

0.0

3.0

721.0

3.0

79.76

5964.0

LITTLEWOODS IRELAND BLANCHARDSTOW

721.0

0.0

0.0

9987

867.0

24.0

0.0

Spain

10001 - 40000

NaN

NaN

NaN

NaN

NaN

...

0.0

2.0

867.0

6.0

132.62

4468.0

RAYMARINE UK LTD PORTSMOUTH

867.0

209.0

0.0

9988

1080.0

64.0

0.0

Sligo Town

10001 - 40000

NaN

NaN

NaN

NaN

NaN

...

0.0

3.0

1080.0

2.0

862.78

3659.0

TAJ LONDON LONDON

1080.0

419.0

0.0

9989

1289.0

34.0

1.0

Rosslare

100000+

NaN

NaN

NaN

NaN

NaN

...

0.0

1.0

1289.0

40.0

327.11

8675.0

AA IRELAND DUBLIN

1289.0

0.0

0.0

9990

1331.0

28.0

1.0

Northern Ireland

10001 - 40000

NaN

NaN

NaN

NaN

NaN

...

0.0

1.0

1331.0

26.0

621.03

5718.0

WESTMEATH STONE CO.WESTMEATH

1331.0

2185.0

0.0

9991

1407.0

60.0

0.0

Lahinch

60001 - 100000

NaN

NaN

NaN

NaN

NaN

...

0.0

4.0

1407.0

40.0

317.29

3068.0

AIR ASTANA ALMATY

1407.0

1752.0

0.0

9992

1626.0

23.0

1.0

Adare

60001 - 100000

NaN

NaN

NaN

NaN

NaN

...

0.0

2.0

1626.0

7.0

170.27

4784.0

EFLOW BARRIER FREE TOL DUBLIN 4

1626.0

0.0

0.0

9993

7461.0

33.0

0.0

Boyle

10001 - 40000

NaN

NaN

NaN

NaN

NaN

...

1.0

4.0

7461.0

1.0

553.73

7216.0

QUALITY DRY CLEANERS DUBLIN 16

7461.0

2064.0

0.0

9994

7618.0

24.0

1.0

Co. Kerry

10001 - 40000

NaN

NaN

NaN

NaN

NaN

...

0.0

4.0

7618.0

0.0

NaN

NaN

NaN

7618.0

36.0

0.0

9995

7935.0

65.0

1.0

999

10001 - 40000

NaN

NaN

NaN

NaN

NaN

...

1.0

4.0

7935.0

56.0

399.95

8062.0

TUNG SHIN HOSPITAL KUALA LUMPUR

7935.0

402.0

0.0

9996

8099.0

28.0

0.0

999

60001 - 100000

NaN

NaN

NaN

NaN

NaN

...

0.0

2.0

8099.0

53.0

430.75

7933.0

THE DIAMOND BOWL & KIDZONWICKLOW TOWN

8099.0

1099.0

0.0

9997

8201.0

55.0

1.0

60001 - 100000

NaN

NaN

NaN

NaN

NaN

...

0.0

3.0

8201.0

0.0

NaN

NaN

NaN

8201.0

440.0

0.0

9998

8297.0

27.0

1.0

County Mayo

40001 - 60000

NaN

NaN

NaN

NaN

NaN

...

0.0

5.0

8297.0

0.0

NaN

NaN

NaN

8297.0

0.0

0.0

9999

8631.0

38.0

0.0

kildare Town

60001 - 100000

NaN

NaN

NaN

NaN

NaN

...

0.0

1.0

8631.0

0.0

NaN

NaN

NaN

8631.0

361.0

0.0

10000

8653.0

36.0

0.0

clare

10001 - 40000

NaN

NaN

NaN

NaN

NaN

...

0.0

3.0

8653.0

26.0

370.74

3548.0

SOL HOUSE ALOHA TORREMOLINOS

8653.0

0.0

0.0

10001

8746.0

33.0

0.0

leitrim

10001 - 40000

NaN

NaN

NaN

NaN

NaN

...

0.0

1.0

8746.0

0.0

NaN

NaN

NaN

8746.0

0.0

0.0

10002

8815.0

65.0

0.0

offaly

10001 - 40000

NaN

NaN

NaN

NaN

NaN

...

0.0

2.0

8815.0

0.0

NaN

NaN

NaN

8815.0

1661.0

0.0

10003

8841.0

43.0

0.0

Kildare town

60001 - 100000

NaN

NaN

NaN

NaN

NaN

...

0.0

3.0

8841.0

14.0

886.64

5551.0

MOONEY BOATS KILLYBEGS

8841.0

0.0

0.0

24 rows × 21 columns

These are the guys with the wrong county names. They are just 24. The BI
assignments should not involve them.

In [64]:

    JoinTest[pd.isnull(JoinTest['Client ID'])]

Out[64]:

Client ID

Age

Gender

County

Income Group

SNo

Population

Density (/ km2)

Rank

Province

...

Held Loan previously

ProductsAmount

Client\_x

Num Transactions

Last TXN Amount

Merchant Code

Last Transaction Narrative

Client\_y

TxnAmount

Loan Flag

10004

NaN

NaN

NaN

Antrim

NaN

2.0

618108.0

202.9

2.0

Ulster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

10005

NaN

NaN

NaN

Antrim

NaN

2.0

618108.0

202.9

2.0

Ulster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

10006

NaN

NaN

NaN

Antrim

NaN

2.0

618108.0

202.9

2.0

Ulster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

10007

NaN

NaN

NaN

Antrim

NaN

2.0

618108.0

202.9

2.0

Ulster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

10008

NaN

NaN

NaN

Antrim

NaN

2.0

618108.0

202.9

2.0

Ulster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

10009

NaN

NaN

NaN

Antrim

NaN

2.0

618108.0

202.9

2.0

Ulster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

10010

NaN

NaN

NaN

Antrim

NaN

2.0

618108.0

202.9

2.0

Ulster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

10011

NaN

NaN

NaN

Antrim

NaN

2.0

618108.0

202.9

2.0

Ulster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

10012

NaN

NaN

NaN

Antrim

NaN

2.0

618108.0

202.9

2.0

Ulster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

10013

NaN

NaN

NaN

Antrim

NaN

2.0

618108.0

202.9

2.0

Ulster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

10014

NaN

NaN

NaN

Antrim

NaN

2.0

618108.0

202.9

2.0

Ulster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

10015

NaN

NaN

NaN

Antrim

NaN

2.0

618108.0

202.9

2.0

Ulster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

10016

NaN

NaN

NaN

Antrim

NaN

2.0

618108.0

202.9

2.0

Ulster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

10017

NaN

NaN

NaN

Antrim

NaN

2.0

618108.0

202.9

2.0

Ulster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

10018

NaN

NaN

NaN

Antrim

NaN

2.0

618108.0

202.9

2.0

Ulster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

10019

NaN

NaN

NaN

Antrim

NaN

2.0

618108.0

202.9

2.0

Ulster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

10020

NaN

NaN

NaN

Antrim

NaN

2.0

618108.0

202.9

2.0

Ulster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

10021

NaN

NaN

NaN

Antrim

NaN

2.0

618108.0

202.9

2.0

Ulster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

10022

NaN

NaN

NaN

Antrim

NaN

2.0

618108.0

202.9

2.0

Ulster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

10023

NaN

NaN

NaN

Antrim

NaN

2.0

618108.0

202.9

2.0

Ulster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

10024

NaN

NaN

NaN

Antrim

NaN

2.0

618108.0

202.9

2.0

Ulster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

10025

NaN

NaN

NaN

Antrim

NaN

2.0

618108.0

202.9

2.0

Ulster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

10026

NaN

NaN

NaN

Antrim

NaN

2.0

618108.0

202.9

2.0

Ulster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

10027

NaN

NaN

NaN

Antrim

NaN

2.0

618108.0

202.9

2.0

Ulster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

10028

NaN

NaN

NaN

Antrim

NaN

2.0

618108.0

202.9

2.0

Ulster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

10029

NaN

NaN

NaN

Antrim

NaN

2.0

618108.0

202.9

2.0

Ulster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

10030

NaN

NaN

NaN

Antrim

NaN

2.0

618108.0

202.9

2.0

Ulster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

10031

NaN

NaN

NaN

Antrim

NaN

2.0

618108.0

202.9

2.0

Ulster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

10032

NaN

NaN

NaN

Antrim

NaN

2.0

618108.0

202.9

2.0

Ulster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

10033

NaN

NaN

NaN

Antrim

NaN

2.0

618108.0

202.9

2.0

Ulster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

...

...

...

...

...

...

...

...

...

...

...

...

...

...

...

...

...

...

...

...

...

...

28229

NaN

NaN

NaN

South Dublin

NaN

37.0

265205.0

1190.6

6.0

Leinster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

28230

NaN

NaN

NaN

South Dublin

NaN

37.0

265205.0

1190.6

6.0

Leinster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

28231

NaN

NaN

NaN

South Dublin

NaN

37.0

265205.0

1190.6

6.0

Leinster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

28232

NaN

NaN

NaN

South Dublin

NaN

37.0

265205.0

1190.6

6.0

Leinster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

28233

NaN

NaN

NaN

South Dublin

NaN

37.0

265205.0

1190.6

6.0

Leinster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

28234

NaN

NaN

NaN

South Dublin

NaN

37.0

265205.0

1190.6

6.0

Leinster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

28235

NaN

NaN

NaN

South Dublin

NaN

37.0

265205.0

1190.6

6.0

Leinster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

28236

NaN

NaN

NaN

South Dublin

NaN

37.0

265205.0

1190.6

6.0

Leinster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

28237

NaN

NaN

NaN

South Dublin

NaN

37.0

265205.0

1190.6

6.0

Leinster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

28238

NaN

NaN

NaN

South Dublin

NaN

37.0

265205.0

1190.6

6.0

Leinster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

28239

NaN

NaN

NaN

South Dublin

NaN

37.0

265205.0

1190.6

6.0

Leinster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

28240

NaN

NaN

NaN

South Dublin

NaN

37.0

265205.0

1190.6

6.0

Leinster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

28241

NaN

NaN

NaN

South Dublin

NaN

37.0

265205.0

1190.6

6.0

Leinster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

28242

NaN

NaN

NaN

South Dublin

NaN

37.0

265205.0

1190.6

6.0

Leinster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

28243

NaN

NaN

NaN

South Dublin

NaN

37.0

265205.0

1190.6

6.0

Leinster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

28244

NaN

NaN

NaN

South Dublin

NaN

37.0

265205.0

1190.6

6.0

Leinster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

28245

NaN

NaN

NaN

South Dublin

NaN

37.0

265205.0

1190.6

6.0

Leinster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

28246

NaN

NaN

NaN

South Dublin

NaN

37.0

265205.0

1190.6

6.0

Leinster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

28247

NaN

NaN

NaN

South Dublin

NaN

37.0

265205.0

1190.6

6.0

Leinster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

28248

NaN

NaN

NaN

South Dublin

NaN

37.0

265205.0

1190.6

6.0

Leinster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

28249

NaN

NaN

NaN

South Dublin

NaN

37.0

265205.0

1190.6

6.0

Leinster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

28250

NaN

NaN

NaN

South Dublin

NaN

37.0

265205.0

1190.6

6.0

Leinster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

28251

NaN

NaN

NaN

South Dublin

NaN

37.0

265205.0

1190.6

6.0

Leinster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

28252

NaN

NaN

NaN

South Dublin

NaN

37.0

265205.0

1190.6

6.0

Leinster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

28253

NaN

NaN

NaN

South Dublin

NaN

37.0

265205.0

1190.6

6.0

Leinster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

28254

NaN

NaN

NaN

South Dublin

NaN

37.0

265205.0

1190.6

6.0

Leinster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

28255

NaN

NaN

NaN

South Dublin

NaN

37.0

265205.0

1190.6

6.0

Leinster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

28256

NaN

NaN

NaN

South Dublin

NaN

37.0

265205.0

1190.6

6.0

Leinster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

28257

NaN

NaN

NaN

South Dublin

NaN

37.0

265205.0

1190.6

6.0

Leinster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

28258

NaN

NaN

NaN

South Dublin

NaN

37.0

265205.0

1190.6

6.0

Leinster

...

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

18255 rows × 21 columns

Counties without any customers (due to the outer join). Remove them

In [65]:

    JoinTest = JoinTest[np.logical_not(pd.isnull(JoinTest['Age']))]

Business Intelligence:

1.  How Many Customers above 50 years old have taken on a loan?

In [66]:

    over50Loan = JoinTest[(JoinTest["Age"] >= 50) & (JoinTest["Loan Flag"] == 1)]
    under50Loan = JoinTest[(JoinTest["Age"] < 50)& (JoinTest["Loan Flag"] == 1)]
    over50Loan.head()

Out[66]:

Client ID

Age

Gender

County

Income Group

SNo

Population

Density (/ km2)

Rank

Province

...

Held Loan previously

ProductsAmount

Client\_x

Num Transactions

Last TXN Amount

Merchant Code

Last Transaction Narrative

Client\_y

TxnAmount

Loan Flag

131

1026.0

67.0

0.0

Cork

10001 - 40000

4.0

519032.0

69.0

4.0

Munster

...

0.0

4.0

1026.0

31.0

257.81

3502.0

BEST WESTERN YOSEMITE WY MARIPOSA CA

1026.0

8125.0

1.0

754

5586.0

54.0

0.0

Cork

10001 - 40000

4.0

519032.0

69.0

4.0

Munster

...

1.0

1.0

5586.0

32.0

253.93

7339.0

REGISTER MY COMPANY DUBLIN 2

5586.0

2647.0

1.0

904

6733.0

69.0

0.0

Cork

10001 - 40000

4.0

519032.0

69.0

4.0

Munster

...

1.0

3.0

6733.0

100.0

168.33

4723.0

FLYCOUK 0572582989880BERLIN

6733.0

0.0

1.0

1060

7982.0

55.0

1.0

Cork

10001 - 40000

4.0

519032.0

69.0

4.0

Munster

...

0.0

4.0

7982.0

13.0

633.97

3014.0

SAUDI ARABIAN AIRLINES JEDDAH

7982.0

0.0

1.0

1109

8302.0

70.0

0.0

Cork

10001 - 40000

4.0

519032.0

69.0

4.0

Munster

...

1.0

4.0

8302.0

19.0

141.15

5973.0

ANGEL INSPIRATIONS SANDYMOUNT

8302.0

6006.0

1.0

5 rows × 21 columns

In [67]:

    over50LoanEver = JoinTest[(JoinTest["Age"] >= 50) & ((JoinTest["Loan Flag"] == 1) | JoinTest["Held Loan previously"] == 1)]
    under50LoanEver = JoinTest[(JoinTest["Age"] < 50)& ((JoinTest["Loan Flag"] == 1) | JoinTest["Held Loan previously"] == 1)]

In [68]:

    len(over50Loan)

Out[68]:

    100

In [69]:

    len(under50Loan)

Out[69]:

    111

In [70]:

    len(under50LoanEver)

Out[70]:

    1510

In [71]:

    D = {"Age : Under 50": 113, "Age: Over 50": 100}

    plt.bar(range(len(D)), D.values(), align='center')
    plt.xticks(range(len(D)), D.keys())


    #plt.hist(Right.prob, bins, alpha=0.5, normed = True, label='true positives')
    plt.title("Number of people with a Loan")
    plt.ylim([0,150])
    plt.text(-0.1,110,'100 People',rotation=0 )
    plt.savefig('Over50.png', bbox_inches='tight')
    plt.show()

![](data:image/png;base64,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%0A9u3q3bu32rZtq5KSkjLr+/n5lTs/wBN4Uxcqha5du6pDhw56++23XcuCg4Nd7+Y9ceKEvvrqK7fG%0APhe2Q4cOKS0tTZGRkWrXrp02btyoffv2SZI+++wz3XfffSosLLzkWOce9+OPP0qS67XJ88/Ifs3q%0A1avldDp16tQprVu3TnfeeadatWqlzMxMbd68WdLPbwyLiYnR0aNH1b59e61atcp1JpmcnKzbb7/d%0A9UTkYuOdr3379lqxYoXy8/MlSS+//PJFXz/18/PT7bffrldffVVRUVGKiIjQ1q1b9dVXXyk6OvqC%0A9b29vcs8MSjPqVOntHPnTj311FPq3LmzsrOzdeDAATmdzsse41L7/YsvvtCDDz6oHj16qE6dOkpL%0AS1Npaelljw14CmfIqDQSEhK0ZcsW122Hw6GnnnpKMTExuv766xUREeHWuIWFherZs6eKi4uVkJCg%0AJk2aSJKef/55PfHEE66z5rlz5/7qmfE5N954oxITEzV69GiVlpbK399f8+bNU1BQULnz+Omnn9Sn%0ATx8VFBQoNjZWkZGRkqRXXnlFs2bNUmFhoSzL0qxZs9SwYUP16dNHhw8fVt++feV0OtW4ceMybzC7%0A2HhZWVmu+/v27avs7Gz169dPNptNDRo00IwZMy46t06dOumTTz5Ru3bt5O/vr/DwcNWsWbPM2eU5%0Ad999t8aOHaspU6aU+zNLUs2aNfXwww+rZ8+eqlWrloKDg3XrrbcqMzPTtQ/O9+KLL+rVV1913e7Y%0AsaOSkpJ+db+PGjVKs2bN0uuvvy5vb2/deuutrpcegN+SzeL6C2A8h8OhQYMGqUuXLkaOB+Dqccka%0AAAADcIYMAIABOEMGAMAABBkAAAMQZAAADFChH3vKycmryM3jKgQHV1Nu7uV/UxIAz+IY/H0KCfn1%0Ajzhyhgy32O3eFT0FoErjGKx8CDIAAAYgyAAAGIAgAwBgAIIMAIABCDIAAAYgyAAAGIAgAwBgAIIM%0AAIABCDIAAAYgyAAAGIAgAwBgAIIMAIABCDIAAAYgyAAAGIAgAwBggMsK8rZt2+RwOMos++ijj9S/%0Af3/X7eXLl6tXr17q16+fNmzY4NlZAgBQydnLW2HBggVavXq1AgICXMu+/fZbrVixQpZlSZJycnKU%0AnJyslStXqrCwULGxsYqKipKvr++1mzkAAJVIuWfIoaGhmjNnjut2bm6ukpKSNH78eNey7du3q3Xr%0A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%0A)

In [72]:

    D = {"Age : Under 50": len(under50LoanEver), "Age: Over 50": len(over50LoanEver)}

    plt.bar(range(len(D)), D.values(), align='center')
    plt.xticks(range(len(D)), D.keys())


    #plt.hist(Right.prob, bins, alpha=0.5, normed = True, label='true positives')
    plt.title("Number of people with a Loan (Ever)")
    plt.ylim([0,1700])
    plt.text(-0.1,len(over50LoanEver)+30,str(len(over50LoanEver))+' People',rotation=0 )
    plt.savefig('Over50Ever.png', bbox_inches='tight')
    plt.show()

![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAeoAAAFXCAYAAABtOQ2RAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz%0AAAALEgAACxIB0t1+/AAAIABJREFUeJzt3XtYlGXi//EPZxUGD4Vu34qSknWtJfGAugi7WopaGpKi%0AjI55yN3MbMUsyPDQ2sFDYmWrrma54YElte/aVzsteskq5hqVlCu5Yml4QAq0mTGH0/P7o5+z4lka%0A9Aner+vqunzueZ778NDNZ+57hhkvwzAMAQAAU/K+1h0AAAAXRlADAGBiBDUAACZGUAMAYGIENQAA%0AJkZQAwBgYgQ1TKGoqEi//OUv9dZbb9UoX7ZsmVJTUz3WTs+ePfX55597rL6LcTgcGjp0qO699169%0A//77V6XNsxUVFSkyMtLj9T799NPKzc2VJKWlpemLL76QJNlsNr333nseayc1NVXLli3zWH1XYs+e%0APXrqqafc/YiJidH9999f478333zTI21VVVXpD3/4g7777juP1If6xfdadwA4zdvbW7Nnz1anTp3U%0AunXra92dn2zPnj367rvv9OGHH17rrnjcc8895/53bm6uhgwZcg1743nV1dV6+umntWjRInfZyJEj%0ANWbMmDppz8fHRw899JCeeeYZvfLKK3XSBn6+WFHDNBo1aqRRo0bp8ccfV3l5+TmPn726OvO4Z8+e%0ASk9P14ABAxQbG6u33npLTz31lAYMGKCEhAQVFxe7r1u1apUGDhyoe++9V2vWrHGXb9q0SYMHD1Z8%0AfLyGDh2qTz/9VJK0YMECjRkzRv3799fkyZPP6dc//vEPxcfHq3///kpKSlJ+fr7279+vKVOmqLi4%0AWPfff79OnTpV4xqbzaYZM2Zo0KBBuvvuu2v8cv7kk09ktVo1cOBAJSQkaPPmze7H/vznP6tfv37q%0A37+/HnvsMZWUlFyyvjMtWrRIAwcO1P33369HHnmkxn2RflzZde3aVQcOHJAkLVmyRD169HA/PmrU%0AKG3ZssW9cp4/f76OHTumyZMna9euXZKk7OxsDRo0SD169NCUKVNUXV19Tj8+++wzDRs2TIMHD9bv%0Afvc7TZky5bz9vZjz3XdJ+vbbb/XII49oyJAh6tmzp2w2m3ul2rNnTy1YsEBWq1U9evTQnDlzzlv3%0Au+++q5tuukmtWrW6ZD+2bt2q/v37u4+///57de7cWSdOnFBxcbHGjx+vhIQE9e/fX4sXL5b0407H%0Ab3/7W40ePVpxcXE6duyYOnfurH379qmgoOCK7wXqOQMwgW+++cZo3769UVVVZVitVmPWrFmGYRjG%0Aa6+9ZqSkpBiGYRgpKSnGa6+95r7mzOMePXoYzz//vGEYhrFhwwajbdu2xp49ewzDMIxHHnnEWLRo%0Akfu86dOnG4ZhGEePHjW6du1q7N271/jqq6+M++67zygtLTUMwzD27t1rREdHG06n03jllVeMuLg4%0Ao6Ki4px+79u3z/jNb35jHDx40DAMw8jNzTWio6MNu91ufPTRR8a999573vEOHz7cGDt2rFFeXm6c%0AOHHCiIuLMzZt2mQcP37c6N27t/HNN9+4+xgbG2scOnTIWLNmjTFkyBDD6XQahmEYr7zyijF69OiL%0A1nf6vhqGYbz99tvGxIkT3ePIzMw0HnrooXP6lpqaamRkZLjrjY6ONvbv3298//33RpcuXQyXy2UM%0AHz7cePfdd933ND8/333+uHHjjMrKSuPkyZNGdHS0sXPnznPaSE5ONj766CPDMAzD4XAYXbp0MT7/%0A/PNzzjv7Z34593358uXGX/7yF8MwDKO6utp46KGHjGXLlrn7evr/raNHjxq//vWv3XWcacKECcba%0AtWtr9KN79+7GgAEDavxXUFBgVFdX17gHK1euNB5//HHDMAzDZrMZ2dnZhmEYxqlTpwybzWZs2LDB%0A+Oabb4zw8PBz7s2sWbOMl19++Zz+oGFj6xum4u3trblz52rgwIHq3r37FV3bu3dvSdLNN9+s66+/%0AXm3btpUkhYaG6sSJE+7zhg4dKklq1aqVunfvru3bt8vHx0fHjh3TyJEj3ed5eXnp4MGDkqT27dvL%0A1/fc6fLRRx+pa9euuvnmmyVJ3bp1U4sWLfTFF1/Iy8vrov0dMmSI/Pz85Ofnpz59+mjr1q3y9vZW%0ASUmJxo8fX6MfX375pXJycpSQkKAmTZpIkkaMGKHFixe7dx/OV1+bNm3c9WzevFmff/65HnjgAUk/%0Abu/+8MMP5/SrV69eyszMVHx8vI4dO6b77rtPubm5atq0qWJiYuTv73/RcfXr108+Pj5q3Lixbr31%0A1vO+7jpr1izl5ORo8eLF2r9/v06dOqWTJ09etN4zXey+P/jgg/r444/1xhtv6Ouvv9Z//vMf3XXX%0AXe5r7777bkk//vyvu+46nThxwl3Pafv379eIESNqlF1s63vQoEF6++239etf/1rr1q3TE088oZMn%0A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%0A)

In [73]:

    bins = np.linspace(18, 75, 30)

    plt.hist(JoinTest["Age"], bins, alpha=0.8, normed = True, label='Population')

    plt.hist(JoinTest.ix[JoinTest["Loan Flag"] == 1, "Age"], bins, alpha=0.5, normed = True, label='People with a Loan')
    plt.xlabel("Age")
    plt.ylabel("Number of peoples (Normalized - pdf)")

    #plt.hist(Right.prob, bins, alpha=0.5, normed = True, label='true positives')
    plt.axvline(JoinTest["Age"].mean())
    plt.text(JoinTest["Age"].mean()+1,0.030,'Age - Average '+str(np.round(JoinTest["Age"].mean(),2)),rotation=90, )
    plt.legend()
    plt.savefig('AgeDistribution.png', bbox_inches='tight')
    plt.show()

![](data:image/png;base64,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%0AAABJRU5ErkJggg==%0A)

The age is surely a factor. For young people should not be so easy to
obtain a loan.

In [74]:

    female3040Products3 = JoinTest[(JoinTest["Age"] >= 30) & (JoinTest["Age"] <= 40) & (JoinTest["Gender"] == 0)]
    female3040Products3 = female3040Products3[female3040Products3["ProductsAmount"] > 2]

In [75]:

    len(female3040Products3)

Out[75]:

    622

In [76]:

    female3040 = JoinTest[(JoinTest["Age"] >= 30) & (JoinTest["Age"] <= 40) & (JoinTest["Gender"] == 0)]

In [77]:

    bins = np.linspace(1, 5, 10)

    plt.hist(female3040["ProductsAmount"], bins, alpha=0.8, normed = False, label='Female 30-40 y.o.')


    plt.xlabel("Products")
    plt.ylabel("Number of Women 30 - 40 y.o.")
    #plt.ylabel("Number of peoples (Normalized - pdf)")
    #plt.xticks(range(len(D)), [,1,,2,,3,,4,,5])

    plt.title("Number of products own")
    #plt.hist(Right.prob, bins, alpha=0.5, normed = True, label='true positives')
    #plt.axvline(JoinTest["Age"].mean())
    #plt.text(JoinTest["Age"].mean()+1,0.030,'Age - Average '+str(np.round(JoinTest["Age"].mean(),2)),rotation=90, )
    plt.savefig('ProductsDistr.png', bbox_inches='tight')
    plt.show()

![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAfIAAAFlCAYAAAAQ8morAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz%0AAAALEgAACxIB0t1+/AAAIABJREFUeJzt3Xl0FGW+xvGns4EkICABRQEBQUaQRXaCbIJBICRAQggQ%0AroarAiLCFSQgmgw7gqAgyjgyRwwg+7Bc5cqWGWQxbAKCZFDAEQRD2LNgFlL3Dw89RuhUOqY7XfD9%0AnOM5dFV31e9937affqs6VTbDMAwBAABL8irpAgAAQNER5AAAWBhBDgCAhRHkAABYGEEOAICFEeQA%0AAFgYQQ4U0pkzZ/Too49q5cqV+ZYvXLhQsbGxxbafTp066Ztvvim27RUkPT1d/fr1U/fu3fXFF1+4%0AfH8TJ07UvHnzivz6CRMm6MiRI8VYEWB9BDngBC8vL82YMUOnTp0q6VKKxbFjx3Tx4kV99tlnCg4O%0ALulyTO3atUtc+gLIjyAHnFC6dGk999xzevXVV5WdnX3L+tjYWC1cuPC2jzt16qTZs2erZ8+eateu%0AnVauXKlx48apZ8+e6t27t1JSUuyvW7p0qXr16qXu3btr1apV9uXbtm1TRESEwsLC1K9fP3399deS%0ApHnz5mnw4MEKCQnR6NGjb6lry5YtCgsLU0hIiKKionT48GGdPHlS48ePV0pKikJDQ/XLL7/ke010%0AdLTi4+MVHh6up556SnPnzpX065GJ9u3bKyYmRsHBwTp//vxtty/9OuN/5ZVXFBwcrOjoaJ08edK+%0A/d8fefjt48TERIWGhiokJESRkZFKTk7WnDlzdP78eY0ePVqHDh3Spk2b1KtXL/Xu3VsRERHau3fv%0Abcds/vz56tatm0JCQjRixAilpqZqy5YtioqKsj+na9euevfddyVJP//8s9q2bavTp0+rc+fOmjRp%0AksLDw9WlSxd9/vnnt90HUKIMAIVy+vRpo3HjxsaNGzeM/v37G9OnTzcMwzA++ugjY+zYsYZhGMbY%0AsWONjz76yP6a3z7u2LGjMXXqVMMwDOOzzz4z6tWrZxw7dswwDMMYNmyY8cEHH9ifFxcXZxiGYfz8%0A889Gq1atjOPHjxunTp0yevToYVy6dMkwDMM4fvy4ERQUZGRkZBhz5841goODjZycnFvq/v777402%0AbdoYP/74o2EYhrFr1y4jKCjISEtLM7766iuje/fut23vwIEDjeeff97Izs42rl69agQHBxvbtm0z%0ATp8+bdStW9fYu3ev6fanTJlivPbaa0ZeXp5x8eJFo127dsbcuXPt7Tx8+LB9fzcfp6amGk2bNjW+%0A/fZbwzAM44svvjAGDx58y2ueeuop4+uvvzYMwzC+/PJLY968ebe0YdWqVUZkZKSRkZFhGIZhzJ07%0A14iJiTGuX79uPPHEE8bVq1eN06dPG0FBQUZkZKRhGIaxePFiIy4uzt7Obdu2GYZhGP/3f/9ndOjQ%0A4bZ9BZQkn5L+IgFYjZeXl2bOnKlevXqpbdu2Tr326aefliRVq1ZNlSpVUr169SRJ1atX19WrV+3P%0A69evnySpSpUqatu2rXbv3i1vb2+dP39ezz77rP15NptNP/74oySpcePG8vG59X/pr776Sq1atVK1%0AatUkSa1bt1bFihV15MgR2Wy2AuuNjIyUr6+vfH191bVrV+3YsUN16tSRj4+PGjdubLr93bt3a/z4%0A8bLZbKpYsaK6dOli2kcHDhxQnTp19Kc//cneZzf77be6d++u4cOHq3379goKCtLzzz9/y3O2b9+u%0A3r17q0yZMpKkQYMGacGCBfLy8lKbNm20c+dOXblyRZGRkVq+fLnS0tK0bds2/fd//7ckydfXV+3b%0At5ckPfbYY7py5Ypp/YC7cWgdKIKqVasqPj5eY8eO1eXLl+3LbTZbvnO4OTk5+V7n5+dn/7evr6/D%0A7Xt5/ed/TcMw5OPjo7y8PLVu3Vrr1q2z/7dixQrVqVNHkuxh9XvGbc4pG4ah3Nxck1Yq3xcDwzDs%0Adfn5+dnXmW3/t+u9vb0d1nbzVIW3t3e+LxiGYSg5OfmWfYwaNUqffvqpGjRooDVr1igyMlJ5eXkO%0Aty9JeXl59rq6dOmi7du3a8eOHWrbtq2aN2+uLVu26Pjx42revLmkX8foZpvNvvQAJYUgB4romWee%0AUbt27bRo0SL7sgoVKth/VX3p0iXt27evSNv++9//Lkk6e/asdu3apdatW6tVq1bauXOnTpw4IUn6%0A5z//qZ49eyorK6vAbd183enTpyVJu3fv1rlz59SoUSPTOtavX6+8vDxdvXpVGzduVKdOnZza/pNP%0APqlVq1bZt7F161b7627O2iXp4MGDSk1NlSQ1atRIJ06c0HfffSdJ2rp1q8aMGSPp15DPzc1Vbm6u%0AOnXqpMzMTEVFRSkuLk4nTpy45ctJ27ZttWbNGmVmZkqSEhIS1Lx5c/n5+alDhw7avXu3jh07poYN%0A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%0Am2rWrKny5cvr6NGj9vUZGRkqV66c6XYuX84s9tpycvOKfZuulJp6+y88gYFlHa7zBM70s6+Pl0eM%0AS3H0p7vMQMB7AAAKSklEQVTHxVX95soxcff7tjjGxBPen5Jz4+LJnw+OxsRT+tlZxdnXBX0pcOvX%0AnFWrVmn69OmSpJSUFKWnpysoKEhJSUmSpO3bt6tZs2buLAkAAEtz64w8PDxc48aNU1RUlGw2m6ZO%0AnaoKFSrojTfe0OzZs1WrVi0FBwe7syQAACzNrUHu5+ent99++5blixcvdmcZAADcMaz5CwIAACCJ%0AIAcAwNIIcgAALIwgBwDAwghyAAAsjCAHAMDCCHIAACyMIAcAwMIIcgAALIwgBwDAwghyAAAsjCAH%0AAMDCCHIAACyMIAcAwMIIcgAALIwgBwDAwghyAAAsjCAHAMDCCHIAACyMIAcAwMIIcgAALIwgBwDA%0AwghyAAAsjCAHAMDCCHIAACyMIAcAwMIIcgAALIwgBwDAwghyAAAsjCAHAMDCCHIAACyMIAcAwMII%0AcgAALIwgBwDAwghyAAAsjCAHAMDCCHIAACyMIAcAwMIIcgAALIwgBwDAwghyAAAsjCAHAMDCCHIA%0AACyMIAcAwMIIcgAALIwgBwDAwghyAAAsjCAHAMDCCHIAACyMIAcAwMIIcgAALIwgBwDAwghyAAAs%0AjCAHAMDCCHIAACyMIAcAwMIIcgAALIwgBwDAwnxKugBJysvLU3x8vP71r3/Jz89PkydPVo0aNUq6%0ALAAAPJ5HzMi3bNmi7OxsLV++XK+++qqmT59e0iUBAGAJHhHk+/fv15NPPilJaty4sY4cOVLCFQEA%0AYA02wzCMki7i9ddf19NPP6327dtLkjp06KAtW7bIx8cjjvwDAOCxPGJGHhAQoIyMDPvjvLw8QhwA%0AgELwiCB/4okntH37dknSwYMHVbdu3RKuCAAAa/CIQ+s3f7V+/PhxGYahqVOnqnbt2iVdFgAAHs8j%0AghwAABSNRxxaBwAARUOQAwBgYXddkB86dEjR0dG3LN+2bZv69OmjyMhIrVixogQqc56jtnz88cfq%0A3r27oqOjFR0drZMnT5ZAdYWTk5OjMWPGqH///goPD9fWrVvzrbfSuJi1xUrjcuPGDY0bN079+vVT%0AVFSUjh8/nm+9VcbFrB1WGpObLl68qPbt2+vEiRP5lltlTH7LUVusNi69evWy1zpu3Lh869wyLsZd%0A5MMPPzR69OhhRERE5FuenZ1tdO7c2bhy5YqRlZVl9O7d20hNTS2hKgvHUVsMwzBeffVV45tvvimB%0Aqpy3atUqY/LkyYZhGMbly5eN9u3b29dZbVwKaothWGtcNm/ebMTGxhqGYRhfffWVMWTIEPs6K41L%0AQe0wDGuNiWH82vfDhg0znn76aeP777/Pt9wqY3KTo7YYhrXG5ZdffjFCQ0Nvu85d43JXzcirV6+u%0AefPm3bL8xIkTql69uu699175+fmpadOm2rt3bwlUWHiO2iJJR48e1YcffqioqCj95S9/cXNlzuna%0AtateeeUVSZJhGPL29ravs9q4FNQWyVrj0rlzZ02aNEmSdPbsWZUrV86+zkrjUlA7JGuNiSTNmDFD%0A/fr1U+XKlfMtt9KY3OSoLZK1xiU5OVnXr19XTEyMBg0apIMHD9rXuWtc7qogDw4Ovu2FZtLT01W2%0AbFn7Y39/f6Wnp7uzNKc5aoskde/eXfHx8Vq0aJH279+vxMREN1dXeP7+/goICFB6erpGjBihkSNH%0A2tdZbVwKaotkrXGRJB8fH40dO1aTJk1SSEiIfbnVxsVROyRrjcmaNWtUsWJF++Wsf8tqY1JQWyRr%0AjUvp0qU1ePBgLVy4UH/+8581evRo5ebmSnLfuNxVQe7I768sl5GRka/zrcQwDP3Xf/2XKlasKD8/%0AP7Vv317ffvttSZdVoHPnzmnQoEEKDQ3N90FrxXFx1BYrjov066zpiy++0BtvvKHMzExJ1hyX27XD%0AamOyevVq7dq1S9HR0Tp27JjGjh2r1NRUSdYbk4LaYrVxqVmzpnr27CmbzaaaNWuqfPnybh8XglxS%0A7dq19e9//1tXrlxRdna29u3bpyZNmpR0WUWSnp6uHj16KCMjQ4ZhKCkpSQ0aNCjpshy6cOGCYmJi%0ANGbMGIWHh+dbZ7VxKagtVhuXtWvX2g9p3nPPPbLZbPLy+vXjwkrjUlA7rDYmS5Ys0eLFi5WQkKA/%0A/elPmjFjhgIDAyVZa0ykgttitXFZtWqV/Y6dKSkpSk9Pd/u43NUXNN+wYYMyMzMVGRmp2NhYDR48%0AWIZhqE+fPqpSpUpJl+eU37Zl1KhRGjRokPz8/NS6dWv7zWg80YIFC3Tt2jW9//77ev/99yVJERER%0Aun79uuXGxawtVhqXp59+WuPGjdOAAQOUm5ur8ePHa/PmzZb7/8WsHVYak9vhM6zkhYeHa9y4cYqK%0AipLNZtPUqVO1ceNGt44LV3YDAMDCOLQOAICFEeQAAFgYQQ4AgIUR5AAAWBhBDgCAhRHkwB3qzJkz%0AatCggUJDQxUWFqbu3bvrueee088//1yk7a1Zs0axsbFOvy4tLU3Dhg0r0j4BmCPIgTtY5cqVtW7d%0AOq1du1afffaZGjRoYL/2uLtcvXpVycnJbt0ncDchyIG7SLNmzfTDDz+oU6dOGjlypIKDg3Xx4kWt%0AXr1aPXr0UEhIiGJjY+2XlVy7dq2Cg4PVp08f/eMf/7Bvp1OnTjpz5owkKSkpyX473WPHjikiIkIh%0AISEaOHCgfv75Z02ePFnnz5/XSy+9pPT0dL3wwgvq3bu3evfufcutXgE4jyAH7hI5OTnauHGjnnji%0ACUlSu3bt9MUXX+jChQtasGCBEhIStGHDBt1zzz167733lJKSolmzZmnJkiVavnx5vmtGOzJ69GgN%0AGzZMGzZsULdu3bRo0SJNmDBBlStX1vz587V582Y9+OCDWrNmjWbOnKl9+/a5utnAHe+uvkQrcKc7%0Af/68QkNDJUnZ2dlq2LChXn31Ve3cuVONGjWSJO3du1cdO3ZUhQoVJEmRkZEaN26cGjVqpCZNmqhS%0ApUqSpJCQEH311VcO93Xp0iWlpqaqY8eOkqT+/ftLkn3mLklNmjTR7NmzlZKSog4dOuill14q/kYD%0AdxmCHLiD3TxHfjulSpWSJOXl5eVbbhiGcnNzZbPZ8q37/W1zb17d+eYtG319ffOtz8rK0vnz52Wz%0A2ezLHn74YW3cuFFffvmlEhMT9be//U0bN27M9xwAzuHQOnCXa9GihbZt26YrV65IklasWKGWLVuq%0AadOmOnTokFJSUpSXl6fPP//c/poKFSro+++/lyT7ee6yZcvq/vvv186dOyVJ69at07vvvisfHx97%0A2C9evFjz5s3TM888o7i4OF26dElpaWnubC5wxyHIgbtcvXr19OKLLyo6Olpdu3bVtWvXNHLkSFWq%0AVEkTJkzQs88+q/DwcAUEBNhfM2LECE2ZMkV9+vTJd3/lmTNn6r333lNoaKg+//xzvfbaa7rvvvtU%0AtWpVRUdHKywsTKdOnbL/GG748OEqV65cSTQbuGNw9zMAACyMGTkAABZGkAMAYGEEOQAAFkaQAwBg%0AYQQ5AAAWRpADAGBhBDkAABZGkAMAYGH/D2Y2BfqjD/gKAAAAAElFTkSuQmCC%0A)

In [78]:

    malesPreviousLoan = JoinTest[(JoinTest['Gender'] == 1) & (JoinTest["Held Loan previously"] == 1)]
    malesPreviousLoan.head()

Out[78]:

Client ID

Age

Gender

County

Income Group

SNo

Population

Density (/ km2)

Rank

Province

...

Held Loan previously

ProductsAmount

Client\_x

Num Transactions

Last TXN Amount

Merchant Code

Last Transaction Narrative

Client\_y

TxnAmount

Loan Flag

0

1.0

36.0

1.0

Cork

10001 - 40000

4.0

519032.0

69.0

4.0

Munster

...

1.0

4.0

1.0

0.0

NaN

NaN

NaN

1.0

58.0

0.0

3

37.0

21.0

1.0

Cork

10001 - 40000

4.0

519032.0

69.0

4.0

Munster

...

1.0

1.0

37.0

4.0

247.65

4468.0

MARINE SUPPLIERS DUBLIN

37.0

28.0

0.0

11

70.0

42.0

1.0

Cork

10001 - 40000

4.0

519032.0

69.0

4.0

Munster

...

1.0

4.0

70.0

15.0

886.13

5966.0

AVGWELLNES441618501835 NICOSIA

70.0

707.0

0.0

12

74.0

66.0

1.0

Cork

40001 - 60000

4.0

519032.0

69.0

4.0

Munster

...

1.0

5.0

74.0

0.0

NaN

NaN

NaN

74.0

0.0

0.0

14

85.0

28.0

1.0

Cork

10001 - 40000

4.0

519032.0

69.0

4.0

Munster

...

1.0

3.0

85.0

11.0

657.52

3389.0

AVIS BIRMINGHAM E211889941

85.0

0.0

0.0

5 rows × 21 columns

In [79]:

    malesPreviousLoan["Num Transactions"].mean()

Out[79]:

    18.6544

In [80]:

    malesPreviousLoanMean = JoinTest.ix[(JoinTest['Gender'] == 1) & (JoinTest["Held Loan previously"] == 1), "Num Transactions"].mean()
    print(malesPreviousLoanMean)

    18.6544

In [81]:

    femalesPreviousLoanMean = JoinTest.ix[(JoinTest['Gender'] == 0) & (JoinTest["Held Loan previously"] == 1), "Num Transactions"].mean()
    femalesPreviousLoanMean

Out[81]:

    17.207336523125996

In [82]:

    malesNoPreviousLoanMean = JoinTest.ix[(JoinTest['Gender'] == 1) & (JoinTest["Held Loan previously"] == 0), "Num Transactions"].mean()
    malesNoPreviousLoanMean

Out[82]:

    18.636742128545407

In [83]:

    femalesNoPreviousLoanMean = JoinTest.ix[(JoinTest['Gender'] == 0) & (JoinTest["Held Loan previously"] == 0), "Num Transactions"].mean()
    femalesNoPreviousLoanMean

Out[83]:

    18.168717528028438

In [84]:

    keys = ["Males \n Previous Loan","Males \n No Previous Loan", "Females \n Previous Loan", "Females \n No Previous Loan"]
    values = [malesPreviousLoanMean,malesNoPreviousLoanMean,femalesPreviousLoanMean,femalesNoPreviousLoanMean]
    plt.text(-0.15,19,np.round(malesPreviousLoanMean,2),rotation=0 )
    plt.bar(range(len(values)), values, align='center')
    plt.xticks(range(len(keys)), keys)


    #plt.hist(Right.prob, bins, alpha=0.5, normed = True, label='true positives')
    plt.title("Average Number of transaction")
    plt.ylim([0,20])

    plt.savefig('TransactionsLoanMales.png', bbox_inches='tight')

    plt.show()

![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAecAAAFhCAYAAABK5GKRAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz%0AAAALEgAACxIB0t1+/AAAIABJREFUeJzt3Xl4jXf+//FXJII4CUmFr6tBg6LqSwRFp2hM26B20ZA6%0AqXWMsRRVW1FtKgmljIy91ZgoRZq2tGpsLVNNbZ3YVetLCEqqlpwkst6/P/yckSIijeRO8nxcl+vK%0Afe7lvO/zdp9X7vuc+xMHwzAMAQAA0yhT1AUAAICcCGcAAEyGcAYAwGQIZwAATIZwBgDAZAhnAABM%0AhnBGsZeRkaFnnnlGgwYNKupSHkj9+vU1d+7cHI9t2rRJVqu1wJ7DarVq06ZNBba93GRlZWnYsGHy%0A9/fXypUrc8w7ePCgpk2bVih1PKikpCQFBwfbp7t166br168XYUWA5FTUBQB/1JYtW1S/fn0dOXJE%0AJ0+eVJ06dYq6pDyLjIzUM888oxYtWhR1KX/YxYsX9e233youLk6Ojo455v3888+6ePFiEVWWu2vX%0ArunQoUP26c8//7wIqwFuIpxR7K1evVqdOnVSrVq1tGLFCr399tt67bXX1LBhQ/vZ9OrVq7V7927N%0AmzdP27dv16JFi5SRkaHy5ctrwoQJatq0qSIiIhQXF6dLly6pfv36mjhxoqZNm6bLly8rMTFRjz76%0AqObNm6dHHnlEBw8e1PTp05WRkaGaNWvq/Pnzmjhxolq2bHnP7d/NmDFj9Prrr+vzzz9XpUqVcsyL%0AiIjQlStX7Gect09brVY9+eST+v7773X58mUFBwfr8uXL2rNnj1JTUzVv3jzVr19f0s1fXpYuXaob%0AN26oS5cuGjZsmCTphx9+0OzZs5WamioHBweNHDlSfn5+iomJUXR0tFJTU2WxWBQVFZWjrn379mnW%0ArFlKTU1V2bJlNXr0aPn6+mrw4MHKzMxUz549FRERoZo1a0qSLly4oPnz5yspKUmTJk1S9+7dNWPG%0ADLm4uCglJUXR0dGaNWuWDhw4oOTkZBmGoXfeeUfNmjXTxIkTZbFY9OOPP+qXX35R7dq19d5776li%0AxYqaP3++tmzZorJly8rd3V1hYWGqWrWqoqOjtWbNGmVkZOjatWsaMmSIgoKCJElLlizRp59+Kicn%0AJ9WqVUvh4eGaNGmSbty4oW7duikmJkYNGzZUbGysPDw8tGDBAn355ZdydHSUt7e3pk6dKk9PT1mt%0AVvn4+OiHH37QhQsX1KxZM82cOVNlynAxEgXEAIqxn376yWjUqJFx5coV48CBA0bjxo2N3377zYiN%0AjTU6d+5sXy4gIMDYtWuXcerUKaNz587Gb7/9ZhiGYZw4ccL405/+ZCQnJxvz5883/P39jYyMDMMw%0ADCMyMtJYsmSJYRiGkZ2dbQwePNj44IMPjIyMDKNt27bGN998YxiGYcTGxhr169c3vv/++1y3/3v1%0A6tUzLl++bIwdO9YYOXKkYRiG8dVXXxn9+vUzDMMw5s+fb7z11lv25W+f7tevnzFixAjDMAwjLi7O%0AqFevnrFt2zbDMAxjxowZxpQpU+zLDR061MjIyDCSkpKMDh06GN98841x9epV44UXXjDOnj1rGIZh%0A/PLLL0bbtm2Nc+fOGZ988onRokULIykp6Y6af/vtN6N169ZGXFycff+eeuop48yZM8bZs2cNHx+f%0Au/bpk08+Mf7yl78YhmEY33//vdGgQQMjISHBMAzD+OGHH4yRI0caWVlZhmEYxpIlS4yhQ4cahmEY%0AEyZMMAIDA420tDQjPT3d6N69uxEdHW2cP3/e8PX1NdLS0gzDMIwPPvjA2LJli2Gz2YyXXnrJ/vr/%0A5z//sde0detW44UXXjCuXr1qGIZhhIaGGgsXLryj7lt9iY6ONgIDA+29mz9/vjFw4ED76zpq1Cgj%0AKyvLSEpKMp555hkjNjb2rvsO5AdnzijWVq9erWeffVaVK1dW5cqV5eXlpTVr1mjo0KFKS0vToUOH%0AVKFCBf32229q3bq1Vq1apUuXLql///72bTg4OOjMmTOSJB8fHzk53TwsXnnlFe3bt08ffvihTp8+%0ArZ9++klNmjTRiRMnJEnt2rWTJLVq1UqPP/64JGnXrl333H6DBg3uug9vvfWWunXrpnXr1snV1TXP%0A+/78889LkmrUqCFJatOmjSSpZs2a2rNnj325gIAAOTk5yWKxyN/fX999950kKTExUcOHD89R548/%0A/ijp5ufhFovljuc8ePCgatasqSZNmkiSHn/8cfn6+mrPnj1q2bJlnmuvXr26Hn30UUlS06ZNValS%0AJX388cc6e/asdu/erYoVK9qXbdOmjZydnSVJ9erV07Vr11StWjU1aNBAPXr0UNu2bdW2bVu1bt1a%0AkrR48WLt2LFDp0+f1vHjx5WSkiJJio2NVYcOHexXKCZNmiRJSkhIuGuNO3fuVM+ePeXi4iJJCg4O%0A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%0AjIzUunXrlJWVpUOHDtnX+/nnn7Vx40atWrVKH330kbZu3ar/+7//0xdffKFBgwZp9erV8vPzY7S1%0AAkSvzOXnn3/Ocan04sWL2rFjhxISErR69Wr985//1OLFi3X9+nVJUuPGjbVixQqlp6erfPny+vDD%0AD1W3bl3t3btX8fHxevHFF7V8+XJ98MEHioyMzPFcixcvVqtWrRQVFaWQkBBNnz5dNptNe/fu1T/+%0A8Q+9//77cnR0LIJXoXigVw8fZ84FrEuXLpo+fbpq1Kih5s2b2x8vU6aMypYtq7Fjx8rFxUW//PKL%0AfchTSTpx4oTOnz+v/v37S5KuXbum+Ph4TZo0SUuWLNHKlStVu3ZtPffcc4W9SyUWvTKXu10qXb9+%0AvY4cOSKr1SpJyszM1Llz5yRJTz75pCTJzc1NdevWtf+clpamKlWqaMWKFdq8ebMsFkuO/kk3e/j9%0A99/rq6++knSzhxaLRZMnT9bUqVNls9nUtWvXh7q/xRm9evgI5wJWo0YNpaSkKCoqSmPHjtXZs2cl%0AScePH9fWrVu1bt06paamqmfPnrr9FvPatWurbt26ev/99+Xg4KDIyEjVr19fa9as0ciRI/XII49o%0A2rRp2rJli3r06FFUu1ei0Cvzq127tlq2bKmQkBBlZ2dr4cKFqlGjxn3XW758uXx8fBQUFKTvv/9e%0AO3bsuGO7Xbt2VZcuXXT58mWtW7dOly5d0pEjR7RgwQKlpaWpXbt26tatm31oYuSOXhWskrMnJtKp%0AUyd9/vnn8vb2tr/h16pVSxUqVFCfPn0kSZ6enrp06ZJ9nQYNGqh169bq27ev0tPT1bhxY1WrVk2N%0AGzfW0KFDVbFiRbm4uOjZZ58til0qseiVubVv31579uxRUFCQUlJS9Nxzz+X4ozv34ufnp3feeUcb%0AN26Uq6urHB0dlZ6ebp//17/+VW+88YbWrl0rm82mESNGyNPTU4mJierTp4/KlCmjgQMHlqg3+4eN%0AXhUsRggDAMBk+EIYAAAmQzgDAGAyhDMAACZDOP9/ERER8vf3t9+316dPH+3evTvf20tMTNT06dML%0ArsD/b+LEifa/pV1SRUREKCAgIMctFS+99JISEhLytH779u318ssvy2q1ql+/fhoxYsQfuuc4JiZG%0A27Zty/f69/KnP/2pwLdpNhxX5sFxVbyUrK+3/UH9+/dX3759JUknT57UuHHj9Omnn+ZrW56eng/l%0A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%0A)

In [85]:

    femaleNoLoan20 = JoinTest[(JoinTest["Age"] <= 20) & (JoinTest["Held Loan previously"] ==0) & (JoinTest["Gender"] == 0)]
    femaleNoLoan2140 = JoinTest[(JoinTest["Age"] > 20) & (JoinTest["Age"] <= 40) &(JoinTest["Held Loan previously"] ==0) & (JoinTest["Gender"] == 0)]
    femaleNoLoan40 = JoinTest[(JoinTest["Age"] > 40) & (JoinTest["Held Loan previously"] ==0) & (JoinTest["Gender"] == 0)]

In [86]:

    len(femaleNoLoan20)

Out[86]:

    120

In [87]:

    len(femaleNoLoan2140)

Out[87]:

    1534

In [88]:

    len(femaleNoLoan40)

Out[88]:

    2003

In [89]:

    D = {"Age < 20": len(femaleNoLoan20), "21 < Age < 40": len(femaleNoLoan2140), "Age > 40": len(femaleNoLoan40)}

    keys = ["Age < 20", "21 < Age < 40", "Age > 40"]
    values = [len(femaleNoLoan20),len(femaleNoLoan2140),len(femaleNoLoan40)]

    plt.bar(range(len(D)), values, align='center')
    plt.xticks(range(len(D)), keys)


    #plt.hist(Right.prob, bins, alpha=0.5, normed = True, label='true positives')
    plt.title("Females with no previous loans")
    plt.ylim([0,2200])
    #plt.text(-0.05,20,'100 People',rotation=0 )
    plt.text(-0.055,200,len(femaleNoLoan20),rotation=0 )
    plt.text(0.90,len(femaleNoLoan2140)+80,len(femaleNoLoan2140),rotation=0 )
    plt.text(1.90,len(femaleNoLoan40)+80,len(femaleNoLoan40),rotation=0 )
    #plt.text(-0.15,19,np.round(malesPreviousLoanMean,2),rotation=0 )
    #plt.text(-0.15,19,np.round(malesPreviousLoanMean,2),rotation=0 )
    plt.savefig('WomenNoLoans.png', bbox_inches='tight')
    plt.show()

![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAeoAAAFXCAYAAABtOQ2RAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz%0AAAALEgAACxIB0t1+/AAAIABJREFUeJzt3XtcVVXC//Hv4XJQ4ZAyoFNPYlqSqVFeuphiWRlmOpop%0AyUkssZtjXovBFDVFUScv/bIwU+tXJN4yyymnZjKTeQhpHnwStajEWzqWCJpwzAPC+v3Rr1OEF0SQ%0ALXzerxevF3uz9l5rnbPO/rI2m71txhgjAABgSV613QAAAHBmBDUAABZGUAMAYGEENQAAFkZQAwBg%0AYQQ1AAAWRlDjknfttdeqb9++6tevn+dr0qRJNV7vhAkTtGzZshqvR5K2b9+u0aNHS5Kys7M1ZcoU%0ASVJmZqb69OlzUdpQk37bv5p05513avv27TVeD1CdfGq7AUB1eOONNxQUFFTbzagx119/vV588UVJ%0A0q5du/TDDz/Ucouq12/7B6A8ghp1Wm5urmbOnKljx46ptLRUMTExGjhwoDIzMzV//nw1bdpU3377%0ArRo2bKhRo0YpJSVFe/bs0T333KOJEyeqrKxMSUlJ2rZtm1wul4wxmjFjhjp16lSpelwul5599lnt%0A27dPXl5eateunaZPny4vr19PZr3xxhvavn275s6dq5KSEt1yyy2aOHGiBg4cqKysLM2aNUtxcXFK%0ATEzUkiVL9OKLL6qwsFDPPvus+vfvrxMnTmjcuHHavXu33G63ZsyYoc6dO5drX2ZmphYsWKDmzZvr%0A22+/VXFxsaZMmaJbb71VhYWFmjZtmnJycmSz2RQREaHx48fLx6f84WHChAmy2WzKzc1VQUGBunbt%0AqoSEBPn6+qp9+/a66667lJOTo7lz56pRo0anfT2efvpptW3bVsOHD5ckrVixQpmZmYqOjlZiYqLe%0Af//9s7bn2muvVUZGhueXsl+W/fz8zvk6/96qVauUkpIiLy8vBQcHa/LkyWrZsqX27Nmj6dOn68SJ%0AEzp8+LDatGmjF154QX5+frr++uv1+OOPKz09XYcPH9bQoUP1yCOPKC8vT/Hx8Tp69Kgk6fbbb9fY%0AsWOrPnCB3zLAJS4sLMz06dPH/OlPf/J8HTlyxJSUlJjevXubHTt2GGOMOX78uLn33nvN//7v/5ot%0AW7aY6667zuzcudMYY8zw4cPNgw8+aNxut8nPzzft2rUz33//vdm6dasZNWqUKS0tNcYYs3jxYvPE%0AE08YY4yJj483S5cuPWs969atM7GxscYYY06dOmUmTZpk9u7dW679Bw4cMF26dDFlZWVmy5YtpmvX%0Armb8+PHGGGPmzJljXn31VbNlyxZz3333GWOMWbt2rXn88ceNMcbTjy+++MIYY8zrr79uhg4dWuE1%0A+qXcl19+aYwxZtmyZeahhx4yxhjzl7/8xSQmJpqysjLjdrtNbGysWbx4cYV9xMfHm/79+5uioiLj%0AdrvNQw89ZFJSUjzvwbp164wx5qyvR0ZGhunTp49nnwMHDjTp6enl+ne29oSFhZn8/Pxy731+fn6l%0AXmdjjOnRo4fJzs42n332mbn77rs9+1q7dq259957TVlZmZk9e7Z59913jTHGFBcXmz59+pgPP/zQ%0AU98vfd6+fbtp3769OXnypHnppZfM5MmTjTHGuFwuM3bsWHP8+PEK9QNVwYwadcLpTn3v2rVL+/fv%0A18SJEz3rTp48qS+//FJXX321rrzySrVt21aSFBoaKofDIbvdrqCgIPn7++vHH39Uhw4ddNlll2nl%0AypX67rvvlJmZKX9//3L17N2794z1REREaMGCBYqJidFtt92mhx9+WC1atCi3/X/913/pj3/8o7Zv%0A365//etfevzxx/Xqq6/KGKONGzdqyZIlOnTo0Bn73rx5c91www2SpDZt2mjt2rWnLXfFFVfouuuu%0AkyS1bdtW69atkySlpaVpxYoVstlsstvtGjx4sN544w09/vjjFfZx//33e/rfr18/bdy4UUOGDJEk%0Azyz+bK9HdHS03G63tm/froYNG6qgoEBdunTR559/7il7Pu35RadOnc75Ov/Wv/71L/Xu3dszZgYM%0AGKCZM2fqwIEDiouLU3p6upYsWaK9e/fq8OHDOnHihGfbu+66S5LUrl07FRcX68SJE4qIiNDjjz+u%0AQ4cO6bbbbtPTTz8th8NxxvqB80FQo84qLS1VYGCg3nvvPc+6I0eOyOFw6IsvvpDdbi9X/veneiXp%0A008/1cyZMzVs2DDdddddatWqldavX1/pevz8/PTPf/5TmZmZ2rJli4YNG6aEhAT16tWr3D569uyp%0AtLQ0paena/HixXr//fe1YcMGNWjQQKGhoWcNal9fX8/3NptN5gy372/QoMFpy5WVlZUrV1ZWplOn%0ATp12H97e3p7vjTHlTi03atTonK+HzWbTwIED9d5778nX11cDBw6UzWarUH9l2lNcXOz5vnnz5pV6%0A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%0AAwBAHXPOGXVoaKgWLlzoWd6xY4c+/fRTPfTQQ5o4caKKioqUnZ2tDh06yG63y+FwKDQ0VDk5OcrK%0AylJERIQkqXv37srIyKi5ngAAUAedc0YdGRmpAwcOeJbDw8M1aNAgtW/fXosWLdLLL7+sNm3ayOH4%0A9RFd/v7+KioqUlFRkWe9v7+/Cgsr9/i8Jk0aycfH+3z7Ap39UWkAfsbnBBfqYo6h877qu2fPngoM%0ADPR8n5iYqM6dO8vlcnnKuFwuORwOBQQEeNa7XC7Pdudy9OiJ820WxLOEgcric4ILVd1jqFqfRz18%0A+HBlZ2dLkjIyMtSuXTuFh4crKytLbrdbhYWFys3NVVhYmDp27KjNmzdLktLS0tSpU6cqdgEAgPrp%0AvGfUzz33nBITE+Xr66vg4GAlJiYqICBAMTExcjqdMsZo3Lhx8vPzU3R0tOLj4xUdHS1fX1/Nmzev%0AJvoAAECdZTPGmNpuxO9xWqpqOPVdc7jhSd1SGzc8YQzVLdU9hqr11DcAALh4CGoAACyMoAYAwMII%0AagAALIygBgDAwghqAAAsjKAGAMDCCGoAACyMoAYAwMIIagAALIygBgDAwghqAAAsjKAGAMDCCGoA%0AACyMoAYAwMIIagAALIygBgDAwghqAAAsjKAGAMDCCGoAACyMoAYAwMIIagAALIygBgDAwghqAAAs%0AjKAGAMDCCGoAACzMpzKFtm3bprlz5yolJUVfffWVEhMT5e3tLbvdrjlz5ig4OFgzZszQ1q1b5e/v%0AL0lKTk6Wr6+v4uLilJ+fL39/f82ZM0dBQUE12iEAAOqSc86olyxZooSEBLndbknSzJkzNXnyZKWk%0ApKhnz55asmSJJGnnzp1aunSpUlJSlJKSIofDoRUrVigsLEypqanq37+/kpOTa7Y3AADUMecM6tDQ%0AUC1cuNCzPH/+fF133XWSpNLSUvn5+amsrEz79u3TlClTNHjwYL399tuSpKysLEVEREiSunfvroyM%0AjJroAwAAddY5T31HRkbqwIEDnuWmTZtKkrZu3aq33npLy5cv14kTJzRkyBANGzZMpaWlGjp0qNq3%0Ab6+ioiI5HA5Jkr+/vwoLCyvVqCZNGsnHx7sq/an3QkIctd0EwPL4nOBCXcwxVKm/Uf/ehg0btGjR%0AIr366qsKCgryhHPDhg0lSbfeeqtycnIUEBAgl8slSXK5XAoMDKzU/o8ePVGVZtV7ISEO5eVV7pch%0A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%0ACyOoAQCwsP8HOqDozUR/PUQAAAAASUVORK5CYII=%0A)

Predictive modelling

Now we will try to predict if a customer is going to ask for a loan.
Before proceeding we have to clean and analyze the dataset

Let's create a copy of the data.

In [394]:

    JoinTraining = JoinTest.copy()
    JoinTraining.head()

Out[394]:

Client ID

Age

Gender

County

Income Group

SNo

Population

Density (/ km2)

Rank

Province

...

Held Loan previously

ProductsAmount

Client\_x

Num Transactions

Last TXN Amount

Merchant Code

Last Transaction Narrative

Client\_y

TxnAmount

Loan Flag

0

1.0

36.0

1.0

Cork

10001 - 40000

4.0

519032.0

69.0

4.0

Munster

...

1.0

4.0

1.0

0.0

NaN

NaN

NaN

1.0

58.0

0.0

1

9.0

24.0

0.0

Cork

60001 - 100000

4.0

519032.0

69.0

4.0

Munster

...

0.0

4.0

9.0

5.0

45.00

5972.0

DUBLIN MINT OFFICE LONDON

9.0

22.0

0.0

2

24.0

57.0

1.0

Cork

10001 - 40000

4.0

519032.0

69.0

4.0

Munster

...

0.0

2.0

24.0

11.0

890.00

5192.0

EASONS KILKENNY CO DUBLIN

24.0

0.0

0.0

3

37.0

21.0

1.0

Cork

10001 - 40000

4.0

519032.0

69.0

4.0

Munster

...

1.0

1.0

37.0

4.0

247.65

4468.0

MARINE SUPPLIERS DUBLIN

37.0

28.0

0.0

4

38.0

60.0

0.0

Cork

100000+

4.0

519032.0

69.0

4.0

Munster

...

0.0

4.0

38.0

9.0

220.27

3420.0

ENTERPRISE GUERIN RENT FARO

38.0

0.0

0.0

5 rows × 21 columns

Delete what would hardly help us in the classification task. We will
take into account some of these features later (if we have time)

In [395]:

    JoinTraining = JoinTraining[JoinTraining.columns.difference(["SNo","Client_x","Client_y","Last Transaction Narrative","Province","Merchant Code"])]

In [396]:

    JoinTraining[JoinTraining["Population"].isnull()]

Out[396]:

Age

Change

Client ID

County

Density (/ km2)

Gender

Held Loan previously

Income Group

Last TXN Amount

Loan Flag

Num Transactions

Population

ProductsAmount

Rank

TxnAmount

9980

65.0

NaN

267.0

Navan

NaN

1.0

0.0

10001 - 40000

NaN

0.0

0.0

NaN

2.0

NaN

864.0

9981

37.0

NaN

323.0

Borris

NaN

1.0

0.0

10001 - 40000

740.69

0.0

17.0

NaN

2.0

NaN

1418.0

9982

66.0

NaN

381.0

Sandyford

NaN

1.0

0.0

10001 - 40000

790.51

0.0

1.0

NaN

5.0

NaN

0.0

9983

43.0

NaN

524.0

Maynooth

NaN

0.0

1.0

10001 - 40000

616.98

1.0

95.0

NaN

5.0

NaN

93.0

9984

49.0

NaN

569.0

Portlaoise

NaN

1.0

0.0

40001 - 60000

NaN

0.0

0.0

NaN

3.0

NaN

377.0

9985

37.0

NaN

654.0

Trim

NaN

1.0

0.0

100000+

NaN

0.0

0.0

NaN

5.0

NaN

331.0

9986

59.0

NaN

721.0

Ballina

NaN

1.0

0.0

60001 - 100000

79.76

0.0

3.0

NaN

3.0

NaN

0.0

9987

24.0

NaN

867.0

Spain

NaN

0.0

0.0

10001 - 40000

132.62

0.0

6.0

NaN

2.0

NaN

209.0

9988

64.0

NaN

1080.0

Sligo Town

NaN

0.0

0.0

10001 - 40000

862.78

0.0

2.0

NaN

3.0

NaN

419.0

9989

34.0

NaN

1289.0

Rosslare

NaN

1.0

0.0

100000+

327.11

0.0

40.0

NaN

1.0

NaN

0.0

9990

28.0

NaN

1331.0

Northern Ireland

NaN

1.0

0.0

10001 - 40000

621.03

0.0

26.0

NaN

1.0

NaN

2185.0

9991

60.0

NaN

1407.0

Lahinch

NaN

0.0

0.0

60001 - 100000

317.29

0.0

40.0

NaN

4.0

NaN

1752.0

9992

23.0

NaN

1626.0

Adare

NaN

1.0

0.0

60001 - 100000

170.27

0.0

7.0

NaN

2.0

NaN

0.0

9993

33.0

NaN

7461.0

Boyle

NaN

0.0

1.0

10001 - 40000

553.73

0.0

1.0

NaN

4.0

NaN

2064.0

9994

24.0

NaN

7618.0

Co. Kerry

NaN

1.0

0.0

10001 - 40000

NaN

0.0

0.0

NaN

4.0

NaN

36.0

9995

65.0

NaN

7935.0

999

NaN

1.0

1.0

10001 - 40000

399.95

0.0

56.0

NaN

4.0

NaN

402.0

9996

28.0

NaN

8099.0

999

NaN

0.0

0.0

60001 - 100000

430.75

0.0

53.0

NaN

2.0

NaN

1099.0

9997

55.0

NaN

8201.0

NaN

1.0

0.0

60001 - 100000

NaN

0.0

0.0

NaN

3.0

NaN

440.0

9998

27.0

NaN

8297.0

County Mayo

NaN

1.0

0.0

40001 - 60000

NaN

0.0

0.0

NaN

5.0

NaN

0.0

9999

38.0

NaN

8631.0

kildare Town

NaN

0.0

0.0

60001 - 100000

NaN

0.0

0.0

NaN

1.0

NaN

361.0

10000

36.0

NaN

8653.0

clare

NaN

0.0

0.0

10001 - 40000

370.74

0.0

26.0

NaN

3.0

NaN

0.0

10001

33.0

NaN

8746.0

leitrim

NaN

0.0

0.0

10001 - 40000

NaN

0.0

0.0

NaN

1.0

NaN

0.0

10002

65.0

NaN

8815.0

offaly

NaN

0.0

0.0

10001 - 40000

NaN

0.0

0.0

NaN

2.0

NaN

1661.0

10003

43.0

NaN

8841.0

Kildare town

NaN

0.0

0.0

60001 - 100000

886.64

0.0

14.0

NaN

3.0

NaN

0.0

This time we have to remove these people beacuse of the NaN in the
features that we will probably include in our model

In [397]:

    JoinTraining = JoinTraining[np.logical_not(JoinTraining["Population"].isnull())]

In [398]:

    JoinTraining['Income Group'].unique()

Out[398]:

    array([u'10001 - 40000', u'60001 - 100000', u'100000+', u'0 - 10000',
           u'40001 - 60000'], dtype=object)

This feature is categorical but at the same time represents a range
binning over a numerical features. Let's reverse it back to be numerical

In [399]:

    JoinTraining.ix[JoinTraining['Income Group'] == "0 - 10000", "Income Group"] = 1
    JoinTraining.ix[JoinTraining['Income Group'] == "10001 - 40000", "Income Group"] = 2
    JoinTraining.ix[JoinTraining['Income Group'] == "40001 - 60000", "Income Group"] = 3
    JoinTraining.ix[JoinTraining['Income Group'] == "60001 - 100000", "Income Group"] = 4
    JoinTraining.ix[JoinTraining['Income Group'] == "100000+", "Income Group"] = 5
    JoinTraining['Income Group'] = map (lambda x: int(x), JoinTraining['Income Group'])

The field of the last transaction is empty for people with no
transactions. However, this could be usefull. Fill it with 0
(consistent)

In [400]:

    JoinTraining["Last TXN Amount"] = JoinTraining["Last TXN Amount"].fillna(0)

Measure the correlation among features to exclude the most correlated
ones

In [401]:

    corr = JoinTraining[JoinTraining.columns.difference(['County','Client ID'])].corr()

    _ = sns.heatmap(corr, annot = True)
    plt.savefig('Corr.png', bbox_inches='tight')
    plt.show()

![](data:image/png;base64,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%0Aq8vwLo6sDLtdYb6eF6eK66MUY2NjlvovYfnyFfj7L6VNmza0b98OmUxGS2dnli7xK5WVShkxYgQf%0AfPABCoUCPT29Cm1VFnENxmskMTERa2trbt26RXR0NAMGDGDIkCHqhg6gWbNmAFhbW5cufnR0pE+f%0APowbNw5vb+9nNqQA7u7u7Nixg71799KjRw+NY1lZWRqLvU6ePMlHH32kpePcuXMUFRWp5+QAHB0d%0Akclk1K5dGzs7O/T09DA1NaWoqEgt88cff/D777+zZs0a9QgGgIWFhXqYbdmyZfTs2ZOePXsSEhKi%0AMYebkpKCiYkJBoaGGvmxsbYmNTW1QjkbGxtSn9Ihl8vJycnh6JEj6v1NmzbFqXFjbpdNJR0PWcdv%0AU4bx8N5N7kddJjfzScOQnZ6KvlFt9PQNNPKRnpRAXLn1Fe907kbWo2Qe3rtFZkoShzf+wm9ThnHp%0A8B6uhx9n72pfLd++Du6ZSzlvJ+O8nYxdO3dq+OrRY18ZaJbF2trmpeSextrahtu3b/PFF1+go6Oj%0AjoVhuXTWNjYVxsvQwAAbGxsePXXscbyOHD2q3t+0aVMaOzlx5/ZtLC0tadiwIYWFhQB8/VVPbt2+%0AjUnt2hp2baytn9L9SCtvT2NpWTpy1fPLL7CxtqawsJDW77xDVFRUjZStYh9XXN818vESMs/idel/%0AVjlz9J5cUDLiEqltY6X+bVrPmrz0TBT5Ty6ytYyNuHsiAr93v2RJh55c3VE64pufXtqG1HunGb8H%0AbeCPuxc5aavLAxMpRbpPbNRUO5KVlcXIHweza/dudu0Iwa1Leyyt66llrUz0ycpXUKCs+BXf3Vq9%0Axa4L8Rr73nOywNKktO1U6ZlyKyaBpvVMNfL7snHSOh8epSCXW1FSUoKhgSFr1/xGSPBWpk6ZTHx8%0APPXr1+fSpUscP/63Oo0gCEgkkhfeeFQGiY6kytubwhvZwcjNzSUkJIRu3brh4OCAi4sLGzZs4M8/%0A/+Tzzz+nfv36Faa7efMmeXl5rF69Gh8fH+bMmaNxXCKRqDsd3bp14/Tp04SFhWl1MMzMzNTDayUl%0AJWRmZqpHM8rj6urK8uXLWbp0qXqYUCJ5fnB/+eUXLly4wB9//KGlMzs7W71v9OjR6sVDwcHBXL16%0AVT0isy0kBFdXVy3dHTt2JDIyktjYWC05V1dXdu3ahUqlIjs7m0MHD9K5c2ekUimzZs3i8uXLANy5%0A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%0AyG6ljuSHmBoaPMnHyIDM7ByysnPUXpM4T58MQORF9RMj7yU84GFqOlPne5D06DG9u1jw7acfl+mz%0A0qfy60JV64yx8ZO+say2Vp1xlCUmoq+vj5WVFfPmO6Gvr09fqz64ODurDZIrsj8xUaZmj7GJCTdv%0A3qBVq1bk5GQT8eefWL/+OpcuXSInJ4fklBRiYmJo2rQpmZmZ2NrZYW5ujuPs2dXuG1pa1K9fn+/m%0AzOHu3btkZWWpDvAqpir9FjzpT8aMHk1qairLlis3wWrZsiWffPIJw4cPR19fn8aNGxMQEEB18MK/%0AIgGIjIxk/PjxqmvPnj0AHD9+nPj4ePz9/fH19WX9+vWkp6er0q1fv56RI0fi5+fH0KFDK23Q8uXL%0AcXBwYOvWrTg5OTF//nwUCgWpqals2rSJ4OBgCgsLVdNlmZmZ/PTTT/z4449s2LBBI7/ExES+/vpr%0AQkJCyM7OJiwsDIA2bdoQEBCAEIJ9+/axrWhKLCwsjLi4OIYNG0ZoaCgAe/bsYdSoUao8AwMDadas%0AGQEBAWzcuJHVq1fz6NGjMv2JjY0lNTWV9evXs2rVKgoLC1W/NW7cmN69e3Py5EkKCwsJDw+vMFYK%0AhaLM+9olvpdWiKfLXL12jYmfTGXMaDsGDuivun/xUhQ5ubm0bNFCI71OqcquKGfnTR1t7TLt1C46%0ArS8hIYGGDRuyefNmhr31FpcuXeLq1atqsjNnzaJBgwZ8NWMGogKfy9JT0k4DAwPCDh3Cz9eXBQsX%0Aci8mBrlczrFjx+hjZcXQN9+kT58+ak9COqVOIhSKcvysxImF/xainDqgU+q7+srKlZX/w4cppKam%0AMqLEQ0XJOvZPyq2yKMopG+1Klo1cXkjE+ShWOX1DkM9S0jIy+X5j+f8cKlsX/mmdeV5xFAoFzZo1%0AY7W3N0ZGRqpyjoj4k4KCgkrZX2bb1tahUaNGeHuv5tdff8Xe3o5r165iZGSErq4ucrmc48ePA+Dn%0A50efPn2YNWtWtfsmFAqOHTvGF19+yTZ/fwDmz5+vJleVfqsYAwMDDoWF4evnx8IFC4i5d48///yT%0A0NBQjh8/zsmTJxk8eDDz5s3TyKdKvAAzGE+t8X379sXPz091jRw5EoAbN24QHR3N+PHj+fTTT1X/%0APIq5d++e6nVCr169Km3Q7du3ee211wCwtLREJpOhra2Nrq4us2bNYv78+chkMtXI1MJCuZ2vmZkZ%0A+fn5GvmZmZmppsN69uzJ3bt3AWjdurXKj/v37zNx4kQmTpxIamoqMTEx2NnZsXPnTqKiomjdurXa%0A4r2SNjZq1Ii2bdsSFxenprd4mrN9+/aMHj2aWbNm4erqqlGJ7ezs2L59O+Hh4bz++uvo6emVGxsz%0AMzOSUx6q/k5KTkZfX58G9es/kTE1rVBm/8FQpn4xg69nfMmUyZMAWPfjT9iN/ZivZilnnFIelkif%0AlIS+vj71G6gvnDMzNSUlJaVMOTMzM1JKvLdPSkrCxMQE46KZrPeKBmuWFhY0atyYK1eUT5anT59G%0AS0uLLp07471qFbq6upiamZWpp0H9+kXxSNHQk5GRweEjR1T3LS0t6dihA2t9fDgeHk5eXh7Hjh4l%0AJSWF999/nxs3/iYuLk5pf311P03NyvGzvuZCwn+TOwY6nG2py9mWuuzcsUPN5mSVzfXV0piamlVK%0ArjSmpmbcvHmTkSNHoq2trVYmKpkqlNuzYmZsSPKjJ9P7iSmP0G/UkAaV3LrZ2KApQ/r1oVHDBujp%0A1uHdwf25eO2mmswPP6xjtL09o+3t2bFje6XqwrPUmYcPH2nIVWccDx85wvnz57G3t2f7jh3cunWL%0AY8eOqeQbN26MtrY22iX+eVdkv9KeJ207uUinQqGgfoMG/PLrrwQFBTN6zBjS09Np0aIFRkZGdOzY%0AUWX3+++/z983bmBgYFCtvimEoHv37rRq1Url262bN8nNzVWlqUq/lZGRwZHDh1X3LS0t6dCxIzdv%0A3eL4sWMMHjwYAwMDtLW1+fjjjzl9+jQSSqr8GNamTRusrKzw8/Nj8+bNvP3227Qo8eTbtm1bLly4%0AAKCabagMbdu25a+//gLg2rVrGBoacv36dcLCwli9ejUuLi4oFArVP3Ctp+w7n5iYqFqkdv78edq1%0AawegalBt2rShXbt2+Pr64ufnxwcffEDHjh155ZVXEELwyy+/YFdq9XJJGzMzM7lx4wbm5ubo6emp%0AdBU/lf/9999kZWWxYcMGPD09cXNzU8vr1VdfJS4ujpCQEGxtbSv05Y033iDq8hViYmMBCA7ZzqCB%0A6u+MrftalSsTGnYYzxVe/LRuDe+8PVyV5stpnxHsvxUTYyPcXReqpQ8JDsbGxkbDFmtra6KiooiJ%0AidGQs7GxYefOncjlctLT0zl44ACDBg2iubk5lpaW7P7jDwA6WljwMCWFZs2aERsby8xZs+jevTuO%0A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%0A)

The information related to the counties are quite correlated. Let's keep
just Density that is the most important for our Loan Flag

In [402]:

    JoinTraining = JoinTraining[JoinTraining.columns.difference(["Rank","Population","Change"])]

In [403]:

    corr = JoinTraining[JoinTraining.columns.difference(['County','Client ID'])].corr()

    _ = sns.heatmap(corr, annot = True)
    plt.savefig('Corr2.png', bbox_inches='tight')
    plt.show()

![](data:image/png;base64,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%0AGVnZ0nxPT2mOo+Mz64JDQqWePXtKLVu2lBIS76i15ermLk3+4QfF39/S9LkW51a9pcv7j0p5WdlS%0AxDSP52orKztHZbmdeEeytraWLl2+ImVl50ienp6So+PcZ9aFhIZJPXvaSC1btpQS79xVrL97L0ma%0APWeO1KFDB+nrr79WajMnK1O6c/u2ZG1tLV25dEnKycqUPD3nS3MdHaWcrEzFUpZm/brfpLFjxkgZ%0AaalS8r27Uv/+/aWTJ44r7Z+TlSm5u7lJP/7wg5STlSkFBwVKQwYPlpKT7kk5WZnSEj8/ycPdXcrO%0AyZES7xQd5+UrV6TsnOLjnDtXys7JUVrK0q1bv14aM3aslJ6RISUlJ0v9+/eXTpw8KWXn5EheXl7S%0AMn9/lfayc3Kk7t27S7fi4xV/30ksyscrly9LOdnZkqenZ9FxZ2crlrI069etK4pNerqUnJRUHJsT%0AUk52tuTl5SX5L1um1FbpJTkpSerbt68UtW+flJuZrrTcvR1fdA5fuijlZqZLC+Z7SHMd5zyTJiwk%0ASLIpPp/u3U5QrB/3zdfSgvkeUm5munTrxjWpS5cu0q0b15TaLu/Ynzc+SffuSXNK5G3JNrt37y7F%0A37qltO7fyJ1x48ZJnp6eUnZOjvTPrVtSly5dpH9u3VJqOyk5Werbt6908ODBKrm2lyb+59GVXrSF%0Al6BEqlpycnL47rvvGDNmDGFhYWzYsIFJkybh5uZW066pEHv8BO3atKZpkyYA2A3/lMjde5AkqUKa%0AWdOnMu3HHwB48OAB+fn5GBvXAaBQXkh6RgYAWdnZ1HqlVgV9iqVN27Y0adoUgOG2tuzevVvJp/J0%0A91NSOHjwAMv8l2u0c+b0afZHRzPHcW6F/KoINhMdiF0XzqmwXVXWZkmOx8bStm07mhYfs62tHbt3%0AR6rEpixdSkoKBw/E4L9cNTb79u3FtKEpU6ZOU2s/9ngs7dq2pWnT4lywtSWyVN+UpYmJOcDQoR+j%0Ar6+PiYkJAz7sz65dkUo2Tp8+TXR0NHMdi0Y0Ll2+TK9eNpi8+ioAffr0Jjo6ushWbCxt25U4Tjs7%0AdkeqxqMsXUxMDEOHDlX49OGAAUTuKuq/c+fO8efJk4wcMYIxo0dz6tQpAO4kJpKVlcV8Dw9shw/H%0Aad489sfEFB93U82xiY3VqCntx4APP2RXZKTCj5N//smIkSMZPWaMwo+S+Pr50aN7d3r06KHab7HH%0Aade2zdNz2Ha4Gt80a1Lu3+fAgUOs8F+q1G5aWhrHT5xkwvjxAFiYmxO4YT11Teqq8UHzsT9vfPbu%0A24dpw4ZMmzpVyWbinTtkZWXhMX8+w21tmefkRFpaWrk5UdrvZ82dtLQ0jh8/zrcTJgBgbm7Oxo0b%0AMTExUWrbz9eX7j160LNnT5V4VQU6erqVXrQF8YikFAcOHKBr16506tRJsa5Dhw5s2LCBe/fuMW/e%0APPLy8njllVdwd3ensLCQadOmYWFhwe3bt2nfvj2urq6kpKQwffp0JEnC1NRU0dbJkyfx8/NDT0+P%0Axo0b4+bmxo4dO9i0aRNyuZwffviBbt26VcjXpORkLMzNFX+bm5mRmZVFVlaWYqiwPI2+vj6z5zkT%0AtT+G3jY9+U/xieg4awbfTJhIQFAwjx49ZpGnB/r65adLclIyFuZPh4LNzMzJysxU8qk8namZGd4+%0AvmXa8fPzZeKkSWqHRCtLyGRnAFr16V5lbZYkKTkJc4unfWFmbk6mmtiUpTMzM8PH109t+7a2RcO7%0A27dtU28/KRnz0rlQyn5ZGtVcMufqtWtKNnz8ljBp0kRFe+3btWNjYBAj7O2pW7cuO3bu5P6DBwAk%0AJyUpt6chHmXpkpOSsCjx6MHc3JxrV68CULduXQYPHkzvPn04c/o0P/30E2Hh4Tx69Ahra2vmODpS%0Av359Fi9aRFBgIJ27dCnTl6TkZMxL2VKKTaltT2LzxI8+vXtz+swZfvrpJ8LDwhRxvn79OgcOHGDn%0Ajh3q+03dOZxZgfO8WGNmaoqfz2KVdhNu36Zhw4YEBG7kj6PHyM8v4CuHLxXXgNI+aDr2543Pk0cl%0A20rl7ZN+cpwzh/r167No8WKcnZ3xW7q0WnMnISGBhg0bsjEggD+OHqUgPx8HBwea/uc/Cu2TPtux%0Ac6dKrKoKbSoUKosoMEqRmJhIk+K7AIDvvvuOzMxMUlJSsLCwYOzYsfTs2ZPY2Fi8vb2ZMmUKt27d%0AYu3atRgaGtK3b1/u37/P6tWrGTx4MHZ2dkRGRhIcHIwkScybN4+goCAaNGjAkiVL2LJli6KCXrVq%0A1TP5KpdLatfrlph9XBHNAndX5s2exdSZP7N6zVq+Gf0VM2bPxd3FiZ4f9ODchQv8MGU67dq0waLE%0Af3zqkCS52vV6pWZEV1SnjnNnz5KamsqAgQPL1WoTkoa+0Ct1Iamo7pnta4h5yVwoSyOXq27TK/Fl%0Ay7Nnz5GamsqgEv0yZPBgUpJTGPfttxjWNuSzzz7FwMAAALmk4ThL/TyvLJ06n54cj6/f00KsU+fO%0AdOzYkdjYWD755BP8lixRbJvw3XeEhYXxdombitJtAUhqbD3RqI9N0XH4+T4tljt36qTkB0BQUFEB%0A9mrxKE9p1LVd2reKaEojk8m4c+cOdeoYs2HdbyQk3Gb0N9/QtHET2rRR/jJxWcdeEU1Z8dFEh/bt%0AWVKiD7+bMIE+fftSUFBQrbnzNC51+P3330lISGDsmDE0adqUNm3aAEV9Zj9ihMY+qwpehp+pvvhH%0AUMVYWFiQmJio+HvVqlUEBARQt25dzp49y//+9z9GjRrFihUrePjwIQBNmjTB2NgYPT09TE1NycvL%0A49atW3To0AGAzp07A0UVeUpKCj/99BOjRo3i6NGj3LlzB4A33njjmX21tDBX3A0CpNy/j4mJiWJC%0AVHmao7HHSbl/HwAjIyMGftify1f+5vqNm+Tm5tLzg6Lh2o7t29O8WTPOX7yo1o9VK1cy0t6OkfZ2%0AbN2yhQcl7N1PScHExATDEj4BWFhYVkinjn379jJ48GB0X4ATcOXKFdjb2WFvZ8eWLZuVjjlFccxG%0ASvtYWFpUSPesWFiob7dkvpSlsbSwKJVLKZibmyn+3rtvH0MGf6TUL2lpaQwcOICIsDACNvzOmTNn%0AKSwsxM7Oji2bNcTDSPk4LTX4ZGhkhKWlJQ+Kc/jJNnNzc9LT01mzZo3SkLkkSRjo63P69GkOHjyo%0AtF5HR0dxPmuMjaWl5thYWirHphw/9IuLrMLCQqL37+fjjz9GEypxT1F3npevKc2TkdWhQwYD0KRJ%0AYzq9/TYX4uJUtGUd+/PEpyxK99PatWuRyWR88fnn1Zo7ZsVx+Xjo0OK4NOHtTp24WHz9KywsZH90%0AdJl9VhW8DI9ItMcTLaFPnz7ExsZy9uxZxbr4+HiSkpLo0KED06dPJyAgAFdXVwYMGACAjo6OSjvN%0AmzfnzJkzAFy4cAGAevXqYWFhwcqVKwkICGDChAl07Vr0y4zK/GfZras15y9eJD4hAYDwTZvp1fOD%0ACmv2RUWz+peii19+fj57o6KxfucdGje2IjMzk7PnzgNwOzGRm7f+oXWrt9T68d333xMcGkZwaBjr%0ANwRw4cJ5EuLjAYiIiKCnjY3KPl27dauQTh2nT53i3fesK6Stab7/fiKhYWGEhoWxISCAC+fPE684%0A5nBs1Bxzt27dKqR7Vrp168b5CxeIjy/OhYhN2Nj0rLDGxsaGrdu2IZPJSM/IYM/evfSy6aXY99Sp%0AU7z33ntK7cVdusSUadMpKChAJpORlpbGxO+LRgwCAgI4X/I4wzXHQ5POxsaGrVu3FvmUns7ePXvo%0A1asXderUITQkhP379wNw5fJlLl68yPvdu5Odnc1CLy/F8/zf16/HpmdPLly4oLARHhGh4ktpP0pq%0ASvuxZ+9ehR8hoaEKPy5fucLFixfp/v77AFy7dg0TExMaNWpURr915fyFkufwJnr1LN1v5WtKY9Wo%0AEa1btWJ78TD/w4cPOXfuPG1LjV6Ud+zPE5+yyM7OxmvhQkU/6erpMWjgQMLCw6s1dxpZWdG6dWt2%0AbN/+NC5nzypGLyrSZ4IixCOSUtSpU4dVq1bh4+ODt7c3MpkMPT09Zs+eTbt27XBxcSEvL4/c3Fwc%0AHR01tvPdd98xY8YMIiMjsbKyAoqKCEdHR8aPH48kSdSpU4dFixZx7969SvnaoH593J3mMW3WbAoK%0AZDS2asR8V2fiLl3GxWM+4UEbNWoApk35EQ9PLz61/xwdHR162/yXL0bao6uri5/3Qhb6+JKXl4++%0Avj5Oc2bTuPg4yqJ+/fo4u7gyc8YMCmQFWFlZ4ebuAcCluDjc3VwJDg0rU1ceCQkJvP7665WKWU1S%0Av34DXFzdmDFjOrKComN295gPQFxcHG6uroSGhZWpex4a1K+Pm4sL00vEfL67O3Fxl3B1cyMsNESj%0ABoomDiYm3sbWfgSyggKGD/+Md955Om8hPiGBRqX65f1u3Th16hS29vbI5RK9etnw5RdfFMWjQQNc%0A3dyYMb2oALGyssJj/tN4uLq6EhYWVqbO1s6O24mJ2NnaUiCTMXz4cN555x0AlixdykIvL1atXIme%0Avj6LFi2iXr169OjRg5EjRzL6q6+Qy+W0KP455blz54qOu9jGfA8PJT8a1K+Pm6uriqYoNrYk3r6N%0ArZ1dcWye+rF0yRK8Fi5k5apV6OvpKfyAiuVyg/r1cXdxYtqMWRQUFNDYyor57q7EXbqEi5sH4SFB%0AGjXlscTHm/leCwmP2IxcLufbcd/Qrm1b9bmj5tirIj6aeNJPX40eregnZyenfyV3fP38WODpSXh4%0AOJIk8e2339KuXbsK91lVoE0jEZVFRyo97Vag1eRlpNa0CwoK9F6paReUmF5H9c6rpvDNvlLTLiih%0AKxXWtAtKSLra9ZZCHS26DOrIZTXtghKSrnbdh0pqRoxrEsPataul3eRFkyu9r/lM1Xfg1ATalTkC%0AgUAgEAheikmeosAQCAQCgUDLeBkekYgCQyAQCAQCLUMUGAKBQCAQCKqcl+ERyYt/BAKBQCAQCLQO%0AMYIhEAgEAoGWUdZbWF8URIEhEAgEAoGWIeZgCAQCgUAgqHKqs8CQy+W4uLjw999/U6tWLTw8PBRf%0AnC3JvHnzqFu3LtOnT6+UnRe/RBIIBAKB4CVDR1e30kt5REdHk5+fT2hoKNOmTcPLy0tFExISwtXi%0ALxNXFlFgCAQCgUCgZVTnx85OnTrFBx8UfZPq7bffVnzI7QmnT5/m3Llz2NvbP9cxiAJDIBAIBAIt%0AozoLjMzMTIyNjRV/6xV/ph6Kviq7YsUKnIq/+/I8iDkYLxpa9B5+fe1xBdCu739MNWpV0y4o4Z11%0AuaZdUEJfi779AWCQ/HdNu6Agz1y7ckfb0LLLzguJsbExWVlZir/lcjn6+kXlwJ49e3j8+DHjx4/n%0A/v375Obm0qxZMz799NNntiMKDIFAIBAItIzqfNFW586dOXDgAIMGDeLs2bO0bNlSsc3BwQEHBwcA%0ANm/ezM2bNytVXIAoMAQCgUAg0Dp0qvGLw/369ePo0aOMGDECSZLw9PRkx44dZGdnP/e8i5KIAkMg%0AEAgEAm2jGgsMXV1d3NzclNY1b95cRVfZkYsniAJDIBAIBAJt4yX4FokoMAQCgUAg0DJ0xKvCBQKB%0AQCAQVDnV+Ijk3+LFH4MRCAQCgUCgdYgRDIFAIBAItI2XYARDFBgCgUAgEGgZ1fkejH8LUWC84Bw+%0A8gdLl68kvyCflm++iavTXKVXwJalyc3NxXPhYi7GXUKS5LRv1445s2Zw5+5dfnacp9i/sFDO9Rs3%0A8F28kL69e6n6cPgwy/z9yc/Pp2WLFri4uKj6oEFTWFiIt7c3x2JjKSwsxMHBATtbWwBO/vknfn5+%0AyGQyXnnlFWbNnEn79u2Bonfp+y1ZQl5eHsbGxri7ufG6VWO1MTpy+DD+/svIz8+nRYuWOKvxryK6%0ApKQkHEZ9SWhYOPXq1VPad+vWLRyIiWHpMv+yuqtSfLXOm7sX/ybK59cqb/vIkcMs9/enID+fN1u0%0AwMlZQ2zK0SUlJTHaYRTBoWHUq1ePmzdu4DhntmJ7oVzOjevXWeztQ+8+fYDqy5snbNm6lZiYGPyX%0ALVNan5+fz+TJkxk+fDj9+vWrUJwOnTiN37pg8gsKaPlGEzymTMC4jpGKTpIkHH1W8WbTxoy1HaJY%0AH7xjLxF7YsjLy6dNi2Z4TJlArVoGFbKtjn8jp6vDbmFhIT7e3sTGHqOwsJBRDg7Y2toB8OefJ/H1%0A8aGwsLDoC54zZvLWW28BRef70iV+5Obl8aqxMW7u7ty8eRP/ZcU2WrZUmz9QlEPqdE9yKPbYMUUO%0A2doV+3LypNK1Z+asWYprT0R4OEFBQejp6dGoUSMWLFhA/fr1K9Brz8hLMIJRZol04sQJpkyZorTO%0A29ubzZs3a9xn1KhR3LhxQ2ndjRs3GDVqlIq2e/fuz+JrjfPLL79w/vz5KmmrKo790ePHzHN1x3ex%0AFzs2R2Bl1Ygl/isqrPn1t3XICmVEhAQSERJEbl4ea9f9TvNmzQgPDlQs73e1ZuCH/dUWF48ePcLJ%0A2Rkfb2+2b9tGIysrli5dWmFNREQECQkJbIqIICgwkMDAQC5cuEBBQQEzZ87EycmJ8LAwxo0bh+Pc%0AuQAkJyczZepUHOfMITwsjL59+jDf01N9jB49wtnZicXePmzdth0rq0YsK+VfRXQ7duxg7Jgx3L9/%0AX2m/tLQ0PDzcWejlhVTFr7+2aNWcn/YH0cXuoypt9wmPHz3C1dmZxYu92bx1G1ZWVvgvU41Nebqd%0AO3bwzVjl2DRr3pzg0DDF0rVrNz4cMEBRXFRX3kBRn7h7eOClpk/OnTvHl6NGcebs2QrH6VFqOo4+%0Aq1gybyqRa5fQ2MIc39+CVHQ3EhIZO8udPYdjldZH/XGCwG17WLtgHtt/8SEvL5/ft+yqsH0Vf6o5%0Ap6vT7qbifguP2MTGwCCCAgO5eOECGRkZTJs6lZ+mTCUsPII5jnOZNXMG+fn5JCcnM23qFGbPcSQs%0ALJw+ffvi6uKCs5MT3j4+bNu+HatGjVTyR+GLBt2THIrYtInAoCC1156w8HDGjRvHXEdHAO4kJrJ8%0A+XJ+W7eO8IgILF9/HX//qr+pAIoKjMouWsKLPwbzLzJ+/Hg6dOhQ024oiI09Qbs2bWjapAkAdsM/%0AI3L3HqWLalmaLp06Mf7rsejq6qKnp0ert1py9949JRunzpwhan8M8+b8rMGHWNq1bUvTpk2L2re1%0AJXL37lI+aNbExMQwdOhQ9PX1MTExYcCHH7IrMhIDAwOi9u2jdatWSJJEYmIir9WtC0BUVBTdu3en%0AdevWAAwfPpyZM2ao9e94bCxt27ZT2La1tWP37kiV/3jK0qWkpHDwQAz+y5ertL9v315MG5oyZeo0%0AtfafB5uJDsSuC+dUWOX/MyqL2OOxtGnblibFxzzc1pbdpfquPN39lBQOHjzAMn/V2DzhzOnT7I+O%0AZo7j3KdtVlPeAOzdtw/Thg2ZNnWqii9BwcFMmjiR9u3aVThOR0+fo91bzflPI0sARgzux86YP1Ti%0AFLx9H8P62zDgv92U1m+LPszozwbzmokxurq6OP8wjo/7/LfC9ktT3TldnXZL99uHHw5gV+QuEhIS%0AMDZ+FWtrawDeeOMN6tQx5vy5c0SrOd+79+hB23YlbNjZsTtS1ZfY2FiNOhVfBgwgctcuDAwM2BcV%0ARavWrRXXnrqvvQYUjcTJZDKysrKQy+Xk5ubyyiuvVDiGz0J1fq793+K5HpH4+Pjw119/IZfLGT16%0ANAMHDlRsS0lJYfr06UiShKmpaYXbTExMZM6cORQWFqKjo8PcuXNp1aoVGzduZN++feTk5FCvXj2W%0AL1/Ozp07OXToELm5uSQkJDBu3DiVN4/16dOHjh07kpCQQIsWLZg/fz4rVqzgzJkzZGdnM3/+fI4d%0AO8bOnTvR0dFh0KBBjBw5kkGDBrFt2zaMjIxYu3Ytenp6XLlyhUGDBtGtWzdmz55NYmIihYWFjBkz%0AhkGDBjFq1ChcXFxo3rw5wcHBPHjwgPHjx/Pjjz+SmZlJTk4OU6ZMoUePHgBkZGQwbNgw9u7di56e%0AHosXL6Zt27YMGjSoQrFKSk7GwsJM8be5mRmZWVlkZWUphgrL0rzfrati/d179wgMCsHJ8emwNoCP%0A3zImT/xO7dDjk/bNLSyetm9uTmZmpooPmjRF/ilvu3rtGgAGBgY8fPgQ+xEjSE1NZdHChQDEx8dj%0AaGjIzFmzuHXrFpaWlsyYPl2Df0mYW5gr/jZT4195OjMzM3x8/dS2/2R4d/u2bWq3Pw8hk50BaNWn%0Aekb6kpOSsTB/GnszM3Oy1MSmLJ2pmRnePr5l2vHz82XipEml4l19efPkUck2NX2y0MsLgN9//71M%0An0uSdP8hFg0bPLVl2oDM7ByysnOUHpPMnTQWgONnlT99fevOPR6mpjN+jicpjx7TpV0rpn3zRYXt%0Aq/hTzTldnXaTk5OU+tTM3Jxr167StGlTcnKyiT12jG7vv0/cxYvcvHmD+w8eKM73WbNmEl98vjdt%0A2hQL86c21OUPQHJSkkZdclKSSg5du3oVeHrtGWFvT2pqKgsXLQKgSZMmfPXVV3wydCivvvoqxsbG%0AhDu2E2UAACAASURBVIWFPVMcK4wWjURUlnJLnePHjzNq1CjFsnPnTgAOHTpEYmIiwcHBbNiwgdWr%0AV5Oenq7Yb/Xq1QwePJiAgAD69u1bYYcWLVqEg4MDgYGBODo6MmfOHORyOampqaxfv57w8HAKCwsV%0Aw6GZmZn873//Y9WqVfzyyy8q7SUnJ/Pjjz8SERFBdnY20dHRADRr1oyQkBAkSSIyMpKg4iGy6Oho%0Abt++Tf/+/dm3bx8AO3fuZOjQoYo2Q0NDqV+/PiEhIaxbt44lS5bw6NEjtceTkJBAamoqq1evxtfX%0Al8LCQsW2V199lS5duvDHH39QWFjI4cOHnylWckmudr1uiRe0VERz6fJlRn89nhH2tvT87weK9WfP%0AnSc1NZVBAz7U6IMkL7/9sjRyNdv0SlTgDRo0IDoqioANG3ByduZWfDwymYyDBw8y8fvvCQsN5b33%0A3mOqmrvVItvqH1volfqkcUV1LxOShtzQK/WCn4rq1HHu7FlSU1MZUOLmA6o/b6oSuYbc0K1gbshk%0AhcSePo+v40+E+S8gLSOTpetCKu1PTeV0VdhV12+6unoYGxvj57eEtWvXYmdny46dO3j33XcxMDBQ%0AnO/ffz+RkNAw3nvvPXbuUj+qVzoH5BoeW+rpavClRP41aNCAqOhoNgQE4OzkRPytWxw7dozo6Gj2%0A7t1L9P792PTqxezZs1XaERRR7ghG165d8fN7Wul6e3sDcPXqVeLi4hRzK2QyGXfu3FHobt26hV3x%0AhJnOnTsTHBxcIYdu3LjBu+++C0Dr1q1JSkpCV1cXAwMDpk6dipGREUlJSYpv17dqVfRpY0tLS/Lz%0A81Xae1LtAnTq1Il//vkHKBqCe3Icd+/eZfTo0UDR89v4+HhsbW1xcXGhWbNmvPHGG0oToG7cuMH7%0A778PFH32tnnz5ty+fVvJ7pOhuhYtWmBvb8/UqVORyWQqc1FsbW0JCAhALpfz/vvvU6tWrQrFCcDS%0AwoILF+MUf6fcv4+JiQlGhoYV1uzeu4/5XouYPXM6Hw0coNT+nn1RDBk8CN0yLtwWlpZcuPj0ji0l%0AJUXFh7I0lpaW3H/wQGmbubk5GRkZnPzzT/r07g0U5cJbLVty/do1TE1N6dixo6Jfhw0bxqJFi8jN%0AzaV27dqsXLmCQwcPAZCVlcmbLVqo2DY0VJ6gZ2FpwYWLF8rVveisWrmSw4cOApCVlcWbbz6NzX3F%0AMRsq7WNhYcnFCxfL1alj3769DB48WCWHqitvqgNLs4acv3Jd8Xfyg0eYGNfBqHbtCu1v1qAefbq/%0ApxjtGNL7A1YGbnomH2oqp6varoWlJQ8ePJ3zcb+43+RyOYZGRqxZu1ax7dNhn9C4cWPF+b5r104O%0AHTyEJMl5+OABycnJqjaMlH2xtLDg4gU1vhgZYWlpyYMS809KXnv+PHlSMV+odevWtHzrLa5dv86f%0AJ0/S08aG+g2KRrTs7e2xHT68wvF8Jv4/jGBoolmzZlhbWxMQEMDvv//OwIEDadz46Sz+5s2bc+bM%0AGQDFaENFaN68OX/9H3t3HhZV9T9w/I1sIoiyg6JWirlbahmZhVBppZkLKJlrWZlauWAqsgmiJpvg%0A0uKKCbK44kbumFL6w9wQ9wRRWRIBAYEZuL8/wImRnUAmv+f1PPM8MPO553zuuefeOXPvmbn/938A%0AJCQkYGxszOXLlzl48CABAQG4uLhQXFyseANXU1OrsrzU1FTFJKYzZ87QoUMHAMUB74UXXqBDhw4E%0ABwezadMmhg8fzosvvshzzz2HJEmsWbMG+ydmp5fNMScnh6tXr2JpaYmWlpairkuXLgFw5coVcnNz%0A+emnn1iyZAmenp5KZfXp04fbt28TGRnJyFp2VOvX+nL+wkUSk5IAiIjcxoC33qxxzK8HD7FkmS8/%0ArgwsN7gAiDtzhr6lg71Kc7C25vz58yQmJpaWH4mNjU2NY2xsbNixYwdyuZzs7Gz2R0czYMAA1NXV%0AcXNzU/Sh69ev89etW3Tv3h1bW1vOnj1LcumA9tChQ7Rv356mpQf8r76aSlh4OGHh4QRv2sSFMnVH%0ARkaUy+9xjjWJ+6+b8tVXiomXG4I3ceHCeZIU6xzJWxWs82vW1jWKq8iZuDheebVvuecbqt80hH69%0Ae3D+8jVu3SmZnxS25wC21n1qvPy7/fsSHfM7+QWFSJLEoZOn6d6x/I2lqtJYfbq+67WxsWFn6XZ7%0AmJ1NdPR+bAYMQE1NjenTphIfX/Jh6MCvv6KhoUHHjh0V+/vQoUMJCw9n/PjxtGvXjvj4+H/qiKg8%0Al7J9qGzck30oev/+So89t/76i+7du9O5c2d+O36cvLw8AA4dPEjPnj1r1aY1paauXueHqqjzHAxb%0AW1tOnTrFxx9/TF5eHm+//bbSta8pU6bg5OTE3r17sbS0rLCMzMxMpTkTkyZNYs6cObi4uLBu3Trk%0AcjmLFi2iXbt26OjoMHr0aABMTExIS0urUZ5aWlp4enpy7949evbsia2treLNH0rOgFhbW+Po6Ehh%0AYSE9evRQfBIaOXIkgYGBvPbaa0plOjg44OLigqOjIwUFBUybNg0jIyPGjRuHh4cHrVq1wtS0ZN7D%0Ac889x8qVK9m3bx/FxcV8/fXX5XIcMmQI+/fvx6rMp4OaMDI0xNPNhVlz5iKTyWlj2ZpFC92Jv3QJ%0Ad89FRIRurjQGIHDFKpAk3D0XKcp8qWdPnOfOASAx6TatWllUm8NCDw9mOzkhk8mwtLRkkZcX8fHx%0AeHh4EB4eXmkMlFwvT759G3sHB+QyGSNHjqRPn5KDd4C/P8uWLUMul6OppcXixYsxMzPDzMwMZ2dn%0AZs6YgUwuR19fH59lyyrMz9DQCHePhTg5zUZeWrenV8n6xsfHs9DDg7Dw8CrjnlWGhoa4uXswx8kJ%0AmbxknRd6lmyXS/HxeC70IDQsvMq46iQlJdGqVatyzzdkv6lvRi1b4DVrCjM8/ZDJ5bSxMGex01Qu%0AXr2Bi/+PbF/9fZXLOw4eSNbDHEZOm0txcTFdOjzPnM/Lf6uuphqrT9dHvfb2DiTfTmaUgz0ymVxp%0Au3kvXoLnQg9kMhnGJib4+QegpqbGi506Md/ZmZkzZiKXy9DX18c/IIA7d+7gNHu2om94Lfonl8d9%0AyNDICI+FCyuMs3dw4HZyMg729sjkyrn4BwQojj1ampqKY8/Qjz7i7t27OI4ejZaWFhYWFiwpnddT%0A71RosmZdqUn1/d06FdOvXz9OnDjR2GlUac2aNbRs2bJGZzAKcrKeQkY1I6nX/HLO01BM1WeznqaZ%0AzTo1dgpKfHITGjsFJRqqs6kA0Ey90tgpKBSYqVbfUTXVnLR+6nRqeKmstgpja3cZrSwt6xH1mEnd%0AiR/aamRz584lLS2NH374obFTEQRBEFSE2jMwB+OZH2Co+tmLBju9JgiCIPx3PQOXSP77ayAIgiAI%0Agsp55s9gCIIgCMJ/jbhEIgiCIAhC/RMDDEEQBEEQ6t0zMAdDDDAEQRAEQcWo0g9m1ZUYYAiCIAiC%0AqhGXSARBEARBqHfPwADjv3+RRxAEQRAElSPOYAiCIAiCilF7BiZ5PvP3InnW5OflNnYKCmpScWOn%0AoERSU50dUq5iJwdn63Zu7BSUBOSp1r1RhMpJKnbzDzUVe8tqqqPTIOUWX/+9zss26fBa9UFPgTiD%0AIQiCIAiqRoU+MNWVGGAIgiAIgqoRAwxBEARBEOqbKl3yrSsxwBAEQRAEVfMMDDD++2sgCIIgCILK%0AEWcwBEEQBEHVqNi3d+pCDDAEQRAEQdU8A7+DIQYYgiAIgqBixCRPQRAEQRDqnxhgCKom5vhxAoOC%0AKCyU0dHKCnc3V/T09GoVk5KSwifjxhMRtgUDA4Na1v8by4NWUCgrpKOVFR6uLhXUX3VMSkoKn4yf%0ASMSWUAwMWgKQlZXF4u+XcfPmTfILCpg8aRJDBn9Qr+tfVFSEj68fJ2NjKSqSM27sOBzsR3Ljxk3m%0AzZ+vWL6ouJjr16/j67MMO1tbVq5aRXT0r+jo6NCzZ09mz5qJunbFv+53/HgMK4KCkBUW0sHKClc3%0A93L51SQuJSWFCePGEhoWjoGBATdv3MB5/jylHG9cv84yH19s7eyq2mS1Mn69D3cvXuGA78//uqyY%0AmJjS7VDSD9zdy7dFZTFFRUX4+PiUbqsixo0bh4O9PTdu3GDePOV2uH79Or6+vrxtZ8fG4GB27NiB%0AhoYGBgYGuCxYQJs2bRolHztbW1auXMmhw4cB6NqlC87OzuiU/jJkY7QPQGFhIdOnT2fkyJFoa2sr%0Ayrfq2LHCHB7nERQYWC7ucR6xJ08q8rB3cAAgMTERdzc3srKy0NHRwWvRIp5//nkkSWLlypVER0f/%0As0/Nno22tjZFRUX8/OOPHD12jEePHtH/jTeYPXs2xxX7dP21VVnJd+7g6OjID6tX07Vr18q6dP16%0ABgYYSCro9u3bkr29fWOnoWTTpk2Sg4OD9PHHH0sff/yxtGLFikbJ41FuTqWPO7dvS3379pUuX7ok%0APcrNkby9F0kLnJ1rFRO+ZYtk89ZbUseOHaW7yberrC8/J1vpcfd2otS3b1/pyqWLUn5OtrR4kZe0%0AwHl+rWLCt4Qo6r93O0nx/OTPPpUWL/KS8nOypVs3rkm9e/eWbt24plT2v13/DevXSZMmTpQeZmVK%0AqffuSu+++6506o/fy62358KF0jdffy09ys2RQkM2S0MGD5ZSU+5Jj3JzpAB/f8nL01N6mJtX7pF0%0AO1nq27evFH8pQXqYmyct8vaW5js71zoudEuY9FZpGyUl36mwLo+FntL0r79W/P8F7f7Vw62TrZRw%0A6IRUkJsnRc7y+tfl3UkuWcfLCQnSo7w8ydvbu2Q75OUpHlXFbFi/vmRbZWdLqSkppdvqD6XlH+Xl%0ASZ6eniXbKi9POnL4sDRo0CApPS1NUcbo0aOrrauh8omKipKGDRsmZWdlSXm5udLUr76SVgQFNVo+%0Aj/LypN9jY6UhQ4ZI3bt3l0I2b5b69u0rJVy+LOU9eiR5e3tLzgsWSHmPHik9ku/cqTRu/YYN0sRJ%0Ak6Tshw+llNRU6d1335X+OHVKynv0SBo2bJgUuXWrlPfokfTrgQPSoEGDpNy8PCkkNFQaPGSIlJKa%0AKuU9eiQFBARIXl5eUt6jR9KPP/4ofezoKD3IyJCyMjOlESNGSMHBwQ3aVpkPHkgO9vZSz549pf87%0AfbpcGzYU2d2rdX6oimdgiNTwQkJC+PPPPwkODmbz5s1s2LCBq1ev8ttvvzV2akpif4+lW9eutGvX%0AFgAHe3v27tuHVOa3+6uKSUtL5/DRI6wICqpb/bG/061rF9q1fVz2yPL1VxGTlp7OkSPHWBm0XKnc%0ArKwsfv/jFF9+/jkA5mZmbA7eQAv9FvW6/ocPH2Ho0A/R0NBAX1+fQQPfZc+evUp1nDlzhoMHD7LA%0AueSMxqWEBAYMsEG/eXMA7OxsOXjwYMXt83ssXbp2pW27dgCMtLdn3xP5VReXnpbG0aNHCAxaUel2%0A+PPMGQ4dPMh85wWVxtSWzdRxxK6PIC58T72UFxv7eDuUrGOF26qKmMOHDzN06NAy22oge/ZWsq0W%0AlLSDkbExzvPnKz7ZdunShXv37jVaPm/b2bFxwwY0NTXJzc0l48EDWrRo0Wj5AISEhjJt6lS6d+vG%0A1atXlcq3d3Bg39695ftrbCxdu3WrMO7JPAYOGsTePXtITU3l1q1bDBo0CIA33niDR/n5XL58mYRL%0AlxgwYAD6+voA2NrZcaB0n4ravZvJkyfTtGlTtLS08PXxobi4uEHbynvxYj788EMMWrZEqB2Vv0Qy%0AduxYOnXqxLVr18jJyWH58uW0bt2aVatWcfDgQYqKinB0dGT06NGsW7eOPXv2oKGhQZ8+fXByciIo%0AKIjExEQePHhAZmYmY8aM4ddff+Wvv/5i6dKlvPTSS2zatIndu3ejpqbG+++/z7hx45RyCAkJITg4%0AGG1tbQA0NTUJCAhATU2N5ORkpkyZQsuWLXnzzTfp168fnp6eqKuro62tjaenJ8XFxcycOZPw8HAA%0AHBwc8PPzY/v27dy8eZP79++TnZ3NggUL6NOnT53bKiUlFTMzM8X/Zqam5OTkkJubqzioVhVjamqC%0Av69v3etPTcW8XNm5yvVXEWNqYoK/77Jy5Sbdvo2xsTGbNv/CbydOUlgoY/y4T3iu9GBRX+tfPjcz%0Arl67plSHr38A06ZNVZTXvVs3ftkcwuhRo2jRogVRu3eT/vffFbZPakoq5mbmiv9NTc3IfSK/6uJM%0ATE3x8fWrsPzH/P39mDptWoWnsutqy3Q3ADrZ9auX8lJSUzEz/2cdzczMym+rKmJSUlMxf+K1ctvK%0Az49pZdrBqkMHxWuFhYUsDwzknXfeabR8oORYErplCytXrMDE1BRbW9tGzWfpkiUAbNy4kczMzGpz%0AAEhNSVHeb8rEpaaklMvj2tWrpKamYmJiQpMy35QwMzUlNTWV7t2788svvzB69GhatGjB7qgo/k5P%0AByApMZEbN2+ydt06Hjx4gM1bb9G0adMGa6tt27Yhl8sZMWIEa9as4WkSkzyfkh49euDs7Iy/vz97%0A9uzhjTfeICYmhoiICIqKivDz8+PKlSvs27ePLVu2oKGhwfTp0zly5AgATZs2Ze3atfz0008cO3aM%0AH374ga1bt7Jnzx709PTYu3cvISEhAEycOJE33niDF154QVF/ZmYmhoaGABw4cIDg4GDy8/Pp06cP%0AY8aMIT09na1bt6KlpcXw4cNZtGgRnTt35uDBgyxZsoQ5c+ZUum5NmzYlODiYa9euMWvWLHbt2lXn%0AdpIqubtpE3X1WsXUVXFx9WXXJOZJcrmcO3fuoKurR/D6dSQl3WbCZ5/Rrk1bunT55y6h/3b9K8pN%0Avck/y549e47MzEzef+89xXNDBg8mLTWNyV98gU5THUaMGI6mpmaFdVRWt/oT617TuIqcO3uWzMxM%0ABpXJURVJNegHVcVUvK3+OSCfLW2H9ytoh4yMDGY7OaGnp8fX06c3ej6Oo0czetQoVq5cyWwnJ9at%0AXduo+SjKr+SupepPfH2yuIq4ivKoLD+AJk2aMHjIEFLT0vh88mR0dHQYPmKEYp+Sy+VcOH+elStW%0AIJPJ+Prrr9HR0VEaPJStR7EudWirhIQEIiIjWbd2bYXLNrhnYIDxn1iDLl26AGBubk5BQQF//fUX%0APXr0QF1dHS0tLebOncvNmzfp2bMnmpqaqKmp0adPH66VjkIfL9+8eXM6lH6KadGiBQUFBVy9epW7%0Ad+8yYcIEJkyYQGZmJomJiUr16+rqkpmZCcA777zDpk2bmD59Og8ePADA0tISLS0tANLS0ujcueRN%0A75VXXlHkUFbZHfe110puq2tlZcXflXzyrSlzc3OlMtLS0tDX16dZmdsJ1ySmrizMzZU+vaelpZcr%0AuyYxTzIxMQFg6JDBALRt24aXX3qJC/HxSnH/dv3L5ZaehpmZqeL/6F9/ZcjgD5Q+dWVlZfHee4OI%0ADA9nU/BGXnjhBcWkQYDVq1bhOMoBx1EO7Ni+Xanu9NK6dZ5Yd3NzixrFVeTXX6MZPHiwUo6qyNzC%0AovptVUWMhYXFE/0oTenMVHR0NEMqaIerV68yZswYOnfqhL+fn+KNqzHyuXLlCgmXLwOgpqbGsGHD%0ASEhIaNT2WblqFQ4ODsRfusQfp05VWL5Os2ZKy1hUsk/pNGuGhYWF4uxD2TwsLCz4+/59pWPh49dK%0A9qn3iIiMJHjTJtqX2adMTEwYNGgQWlpa6Orq8s477/AgM7NB2ioqKoqcnBzGjx+Pg4MDaenpzJs/%0An6NHj/JUqKnV/aEiVPsoVIkXXniBS5cuUVxcjEwmY+LEiTz//POcP38euVyOJEmcPn2a559/HijZ%0Aeasqq0OHDgQHB7Np0yaGDx/Oiy++qBQzZswYvL29KSwsBKCoqIi4uDhFuWV3UlNTUy6XHjROnz7N%0Ac889h7a2Nvfv36eoqIjs7GySk5MV8fGlb5JXr15VOgDUhbW1NecvXCAxMQmAiMit2Ni8VeuYutf/%0AGucvXCQxqbTsrVsZ8NaT9Vcf8yTL1q3p3KkTu3bvBuD+/fucO3eermXOXtR03aqKsbGxYcfOncjl%0AcrIfPmR/dDQDbAYolo2Li+PVV19VKi/+0iVmzJqNTCZDLpezbt16Pnj/n0+FU776itCwcELDwtkQ%0AvIkLF86TVDqAjYyM5C0bm3Lr+5q1dY3iKnImLo5XXu1bo9jGZG1tzfnz5xWD+YjISGyeWMeqYmxs%0AbNixY0fJtsrOLtlWA57YVn2V2yEpKYnPJk/m8y++wMnJSemMUGPkc/XaNdxcXXn06BFQMr/gcf9q%0AjHwApn71FeHh4XTt0oXp06crlR8ZEVEuh4ryKBv3ZB7R+/czYMAAzMzMaGNpSfT+/QCcPHGCJk2a%0AYGVlRXx8PDNnzFDsU2vXreP9D0q+Mfb222+zZ88exbE/5vhx3ujXr0Haas6cOUTt2kV4eDjh4eGY%0Ampiw2Nu7wjZoEGpN6v5QEf+JSyRP6ty5M/3798fR0ZHi4mIcHR3p1KkT7733nuK53r178/bbbyve%0A7CvTqVMnrK2tcXR0pLCwkB49epR7ox83bhyhoaFMnDiRJk2akJOTw0svvcTMmTMpKChQivXy8sLT%0A0xNJklBXV8fb2xsTExP69evHyJEjadOmjWKiEUBCQgLjx4/n0aNHeHp6/qt2MTI0ZKG7O7OdnJDJ%0AZVhaWrLI05P4+Et4LFxIeNiWSmPqg5GhIZ7ursxy+g6ZTEYbS0sWeXoQf+kS7gu9iNgSUmlMdQJ8%0AfVi0ZCkRkdsoLi7mi8mf0e2Jr4v92/V3sB9JcvJt7EeNRi6TMXLkCPr06a0oPzEpidatWinV+bq1%0ANXFxcdiPGkVxscSAATZ8MmYMFZ00NjQ0xM3dgzll6l7o6QXApfh4PBd6EBoWXmVcdZKSkmj1RI6q%0AyMjQkIUeHiXbQVa6Hby8iI+Px8PDg/Dw8EpjoGSSXvLt29g7OJRuq5FK85cq2lbr168nPz+f0JAQ%0AQksviWpqabH5l18aJZ8hgwdzOymJj8eMQV1dnfbt2+Pu5tZo7fOk5np6LPTwwGn2bEX5XosWASjl%0AYWhkhMfChRXG2Ts4cDs5GQd7e2RyuVIeS5YuZeHChfz8889oa2uzzMeHJk2a8PrrrxMXF4eDvT3F%0AxcUMGDCATz75BICp06axPCCAESNHIpfLsX7tNSZPnky3bt0arK0ay7MwB0NNquxCm9DggoKCMDY2%0AxtHRscbL5OflNmBGtaNWyVyBxqJKO6RcxU4OztbtXH3QUxSQl9DYKQg1JKnQKXcANRV7y2paD5eX%0AK1L4d3L1QZXQMrasx0zq7j95BkMQBEEQnmkqPo+qJsQAoxFNL53BLgiCIAhKVOiMbF2JAYYgCIIg%0AqBoxwBAEQRAEod6JAYYgCIIgCPVNlSat15UYYAiCIAiCqnkGBhj//TUQBEEQBEHliAGGIAiCIKia%0ABvyp8OLiYlxdXRk1ahRjx44td3uMw4cPM2LECEaNGqW4SWddiEskgiAIgqBqGvASycGDByksLCQs%0ALIyzZ8+yZMkSVq9eDYBMJmPx4sVERkaio6ODo6Mjtra2GBsb17oecQZDEARBEFSMpNakzo/qxMXF%0A0b9/fwBeeuklLl68qHjtxo0btG3blhYtWqClpUXv3r05ffp0ndZBnMEQBEEQBFXTgGcwcnJy0NPT%0AU/yvrq6OXC5HQ0ODnJwcmjdvrnhNV1eXnJycOtUjBhj/NSo0s1jVvkalSvdM0FCx+yWo2r0/vm2m%0AWvdGCbx/srFTUJAf2NjYKSgpHvJtY6egREN1dvMG1ZDHMz09PXJz/7mvVXFxMRoaGhW+lpubqzTg%0AqA3VeocQBEEQBAFJqvujOr169SImJgaAs2fP0rFjR8Vr7du3JzExkczMTAoLC/m///s/Xn755Tqt%0AgziDIQiCIAj/Q9555x1OnDjB6NGjkSQJb29voqKiyMvLY9SoUcydO5dPP/0USZIYMWIEZmZmdapH%0A3K79Pyb/0aPGTkFlqdIlElW7pbSqEZdIKicukVRN1S6RNNTt2nPy6n6s12vWMDnVljiDIQiCIAgq%0A5ln4iCIGGIIgCIKgYoqfgRGGGGAIgiAIgop5FmYviAGGIAiCIKgYcQZDEARBEIR69wyML8TvYAiC%0AIAiCUP/EGYz/oJiYGAKDgigsLKSjlRXu7u5KP/taVUxRURE+Pj6cjI2lqKiIcePG4WBvD8DRY8dw%0AcXHBwtxcUc769evR1dUlLi4O/4AACgoK0NPTw3PhQiwtLRs0n1OnT+Pv749cLkdbW5vv5syhe/fu%0AijILCwuZPn06I0eO5J133lGqKygwkMLCQqw6dqwwn6riHucUe/KkIid7BwcAjh09iouLC+YWFuXa%0AaNbMmVy9ehWdZs1QkyReeeUV+vbt2yBt89j2HTs4fPgwQYGBSs9X1Tb1nc+NGzeYN2+eYvmi4mKu%0AX7+Or68vb9vZsTE4mB07dqChoYGBgQEuCxaU2xZ1NX69D3cvXuGA78/1VuZjMSdiCVi9BplMhlX7%0AF1jo7ISerm6NYmbOdyMp+Y4i7s7dFPq83JOgZYsUz22P2suhY7+xwse71rkdv3KboINnKJQXYWVu%0AgNvQfug11VKK2fJHAhGnrqCmBpaGzXH98HUM9XTIyivAe3csV+5loKOlwYcvW+H42r/72vDx4zGs%0ACApCVlhIBysrXN0q3ucqi8vPz2fpksXEx8cjFRfTrXt3vps7j6ZNm1ZYX0Mdcx5LvnMHR0dHfli9%0Amq5duyJJEitXruTQ4cMAdO3SBWdnZ3Qa6Cuq8GxcImmQMxh//PEHM2bM+FdlhIWFIZPJlJ6bNWsW%0AY8eOxdbWloEDBzJ27Fg8PT05ceIEQ4YMoaCgAIDU1FSGDBlCamoqc+fOZdq0aUrl9OvXr9J6hw4d%0AioeHx7/KvS5Onz7N5cuXq43LyMjA1c0NXx8fdu3cSWtLS5YvX17jmMjISJKSktgaGUnI5s1sj7uK%0AcgAAIABJREFU3ryZCxcuAHDu3DnGjxtHeHi44qGrq0tqaiozZs7Eef58IsLDedvOjkXe3g2aj0wm%0AY86cObi6uhIRHs7kyZNxLvPGdO7cOT4ZO5Y/z54tV5ebqys+vr7s3LULy9aty+VTXdzjnCK3bmVz%0ASEi5Nho3fny5NgI4f/48a9etUzz/6aefNti2ysrKwtPLiyVLlpSbDFZV2zREPu3bt1dqD2tra94b%0ANIi37ez4/fff2bFjB5uCg4kID8fO1hZXN7dy26O2zDu159tDIfR2+OBfl1WRjAeZuCz6Hv/FHkSF%0ABWPZ2oKAVT/VOMbP24PI4DVEBq/Bfe5smjfXw3n2NwBkZWWzcKkfi/2C6jSRLyM3H7cdJ1g2egA7%0AvhmOpUFzAg/EKcVcuvs3wScusmHy+0RO+4i2hvqsOvwnAD77T6GjpcnW6R8RPPkDTlxLJubK7bo0%0AEwAPMjLwcHNj2TIftu3YiaWlJUGB5fe5quLWrV1DkbyILWHhbAmPoCC/gPXr1lW8/g14DAQoKCjA%0Aef58pfefQ4cPczI2lvCwMLZt3Up+fj4hISF1brOakCSpzg9VobKXSH788UeKi4uVnvP19WXTpk0M%0AGzaMCRMmsGnTJlxcXOjXrx/9+/fH29sbmUzGjBkzmDt3ruLXx+Li4tixY0e1dcbFxdGxY0d+//33%0AOt/cpa62bt1KWlpatXGxsbF069qVdu3aAeBgb8/effuUOlVVMYcPH2bo0KFoaGigr6/PoIED2bN3%0AL1DyxnTq9GlGOzoyYeJE4uJKDloHDhygX79+dO5c8iln5MiRzHFyatB8NDU1OfDrr3Tu1AlJkkhO%0ATqZlixaKMkNCQ5k2dSrdu3Ur1z5du3VT1GXv4MC+vXvL7XRVxT2Z08BBg9i7Z4+ijU6fOoXj6NFM%0AnDBB0UZ3kpPJzc1lkZcX9iNH4uLqyqHDhxtsW0X/+ismxsbMmjmzXB+pqm0aKp/Hzpw5w8GDB1lQ%0AOhg0MjbGef58xafLLl26cO/evXI515bN1HHEro8gLnzPvy6rIidPnaZr5xdp16bkLN2o4UPZE31I%0Aqa1qEiOTyXD2XMJ330zF3MwUgOhDRzExNmLW9C/rlNvv1+/QtZUx7Yz0AbB/5UX2nb+pVG+XVsbs%0A/GYEzZtqUSCTk/YwjxY62gAk3L3P4J4voN6kCZoa6vTvaMnB+Ft1ygUg9vdYunTtStvSPjPS3p59%0AT/Sr6uJe7tWLTydPpkmTJqirq/Nipxe5d+9uxfU1cD/2XryYDz/8EIOWLRXPvW1nx8YNG9DU1CQ3%0AN5eMBw9oUeZ41BCK/8VDVTzVSyT79+9n8+bNyOVy1NTUWLFiBQDffvstkiRRUFCAh4cHFy9eJD09%0AnRkzZrBq1aoalT1jxgwcHR2ZMmUKr7/+utJZipkzZxIUFMRrr72GeZnT/0+KiIhg4MCBWFhYsGPH%0ADj755BOSk5OZMWMGFhYWJCcn88EHH3Dt2jUuXbqEjY0NM2fO5NKlS3h6eqKuro62tjaenp4UFxcz%0Ac+ZMwsPDAXBwcMDPz4/t27eTnJzM/fv3uXv3LvPmzcPAwIDjx48THx9Phw4daNWqVaU5pqSmYlZm%0AHczMzMjJySE3N1dxEK8qJiU1VakNzMzMuHrtGgAtWrRg8ODB2NnacubPP/n222+JCA8nMTERHR0d%0A5nz3Hbdu3cLCwgKn2bMbPB9NTU3u37/PqNGjyczM5PulSxVxS5csAWDjRuVfPUxNScG8zM/aVpRP%0AdXGpKSnlcrp29apSG9na2fHnmTN8++23hEdEkJGRQd++fZnv7IyhoSHLvv+ekM2b6dW7d4O0zeNT%0Aujt37uRJlbVNQ26rx3z9/Jg2bZqiPKsOHRSvFRYWsjwwkHfeeYfbx/7dJY0t00vOgnSyq/xs5L+R%0AkpqOuamp4n8zExNycnPJzctTXCapScy2qL2YGBthZ9NfEecw/EMAduzZX7fcsnIxa9FM8b+pvi45%0ABTJyC2RKl0k01ZtwJCGRhTtPoqmuzpRJJfeT6GZpwu5zN+nZ1gyZvIhDlxLRaFL3z5qpKamYm/3T%0AL0xNzcitcJ+rPM7a+nXF8/fu3iVkcwjOLi4Vr38D9uNt27Yhl8sZMWIEa9asUapXU1OT0C1bWLli%0ABSamptja2taqnWpLhU5E1NlTPYNx69YtfvrpJ0JDQ+nQoQO//fYb58+fp2XLlvz888+4urqSl5eH%0Avb09JiYm+Pv717hsTU1NRo0aRWxsLMOHD1d6zczMjG+++QZnZ+dKl8/JySEuLg4bGxuGDx9OaGio%0A4rXbt2+zaNEifvzxR5YvX87cuXOJiIggMjISgAULFuDq6sovv/yCo6MjS0oP8JXR0tJizZo1ODs7%0As2HDBrp160b//v1xcnKqcnABIBVXPD5toq5eo5gnzwoBqJceXPz9/LAr3Wl6vfwyPXv2JDY2Frlc%0AztGjR5n61VeEh4Xx6quvMrP0k3ND5gNgZGTEwQMH2BQcjKubG7cSEyss67HiSvZK9ScOoFXFVZTT%0A4/Xx8/fH1s4OgJd79VK0UfcePfAPCMDExAR1dXWmfPkltxITqywL/l3b1FZDb6uzZ8+SmZnJ+++9%0AVy4uIyODL6dMoVmzZnw9fXptU3/qKm2HMutbk5hNWyL5YuLY+s2tkjce9Sblf0N7QOd2HJnryJcD%0AXmJq8K8UF0vMGtgHNcBx9S5mhh6mb/tWaKr/i34lVdwO6mX6VU3jEi5d4tNPJzFq9CjefPPNiutr%0AoH6ckJBARGQkC6p4n3AcPZrjx49jZ2vL7NKzuA2lWKr7Q1U81QGGkZER3333HfPmzePKlSvI5XLe%0AfPNNevXqxVdffUVgYKDSzlkbycnJrFmzBicnJ5ycnCgqKlJ6/cMPP0RXV7fS62a7du2iuLiYL774%0AAk9PT9LT04mNjQWgTZs2NG/eHH19fYyNjWnZsiXa2tqold77Ii0tTXH54JVXXuHaE5/qQPlHUx7H%0AmpubU1hYWKv1NLew4O+//1b8n5aWhr6+Ps3KTDaqKsbCwoL0J14zMzMjOzubNWvWKOUpSRIampqY%0AmJjQs2dPxenGYcOGceXqVfLz8xssn4cPHyomVD1usxc7duR6BW27PzoaBwcHHBwc2L5tW4V16TRr%0AprSMhbl5pXEWFhb8nZ5e4zbS1NDgzJkzHD16VOl5NTU17t+/X+9tU1cNta0ei46OZsjgweX24atX%0ArzJmzBg6d+qEv58fmpqadV6Hp8Xc3Iz0stsuPR395s2V26qamIQr15AXFdHn5Z71m1tLXf7O+ec+%0AFWkP89DX0UJH6592TbqfzZ+JqYr/h/bqwL3MXLLzC8gpkPHtu32InPYRP0wYSBM1NdqUXm6pqdWr%0AVuE4ygHHUQ7s2L5dqc+kP96XnpgAaW5uUWVc9P79fDXlS6Z//TWTPv2s8vVvoH4cFRVFTk4O48eP%0Ax8HBgbT0dObNn8/Ro0e5cuUKCaVz5NTU1Bg2bBgJCQm1arPaEnMwauHhw4cEBgbi7++Pl5cX2tra%0ASJLEH3/8gampKevWrWPKlCn4+fkBJRuxopFmRQoLC5kxYwbz589nwoQJWFhYKC6/lOXu7s66deuU%0A7nX/WGRkJD/88ANr165l7dq1LFiwgM2bNytyqYqpqaligubp06d57rnn0NbW5v79+xQVFZGdnU1y%0AcrIivqLy1NTUatQxrK2tOX/+PImln+QjIiOxsbGpcYyNjQ07duxALpeTnZ3N/uhoBgwYgK6uLlvC%0Awjh06BAACZcvc/HiRfq9/jq2tracPXuW5Dsls+IPHTpE+/btadq0aYPlo66ujpubG3/+WTIx7fr1%0A6/x165bSt0geGzRwoGKC4aZNm5TqioyIKJdPRTmVjXsyp+j9+xVtFLZli6KNLickcPHiRV7v14+8%0AvDyWLllCVlYWABs2bsTmrbe4cOFCvbdNXTXUtnosLi6OV/v2VSovKSmJzyZP5vMvvsDJyancp1pV%0A9fqrfTh/MYHE2yX7bfj2KAa82a9WMf/35zn69n652uNHbVm3b8WF2+kk3s8GIPL0FWw6tVWK+fvh%0AI+ZGHONBbj4Ae8/fpL1pS1o2a0rk6SusLp3weT/nEdvjrvJejxdqlcOUr74iNCyc0LBwNgRv4sKF%0A8yQ93pciI3mrgn3uNWvrSuMOHjjAsu+XsnLVat577/2q17+B+vGcOXOI2rVLcSwxNTFhsbc3NjY2%0AXL12DTdXVx6V3mwyavduXn311Vq12f+iBpuDceLECaVLFT4+PvTq1YtRo0YpJtekpaVha2vLzJkz%0ACQ0NRS6XM3XqVAD69OnD559/TnBwcLU76NKlS+nduzdvvfUWUDKQGD58OK+99ppSnKGhIXPnzlXU%0A8Vh8fDySJGFlZaV4buDAgSxevJiUlJRq19XLywtPT08kSUJdXR1vb29MTEzo168fI0eOpE2bNopP%0A/5Xp2bMnPj4+WFpa0r59+0rjjAwNWejhwWwnJ2QyGZaWlizy8iI+Ph4PDw/Cw8MrjYGSa/fJt29j%0A7+CAXCZj5MiR9OnTB4DlAQEsWbqUVatXo6Guzvfff4+BgQEGBgY4Ozszc8YMZHI5+vr6+Cxb1uD5%0ABPj7s2zZMuRyOZpaWixevLjaT/CGRkZ4LFyI0+zZirq8FpV8NbBsTlXF2Ts4cDs5GQd7e2RyuXJO%0Ay5ezdMkSVq9ahbqGhqKN3njjDRwdHZkwfjzFxcVYlX4t7ty5cw3SNnXRkNsKIDEpidZPXOJbv349%0A+fn5hIaEEFp69lBTSwuTOq/F02FkaIDngjnMnO+GTCanTetWeLvOIz7hCm6LlxEZvKbSmMcSbyfT%0AyqLyOV91Zaing/uwN3DacgR5UTGWhs3xHN6f+Dt/s3DnCcK+Gkqv58z49M0eTF6/H/Umapg0b4b/%0AxyWXPye92YMFW2MYuWIHkgRfDHiJrq2N656PoSFu7h7McXJCJi/pMws9S/rMpfh4PBd6EBoWXmXc%0AiqBAJAk8F/7zDb6eL73E3Hnzy9XX0P24IkMGD+Z2UhIfjxmDuro67du3x70evg1VFVWarFlX4nbt%0A/zHidu2VE7dr/+8Qt2uvnLhde9X+V27Xnni/7t9kbGdU/jdIGoP4oS1BEARBUDGVTUT/LxEDDEEQ%0ABEFQMf/94YUYYAiCIAiCylGlr5vWlRhgCIIgCIKKeQaukKjuT4ULgiAIgvDfJc5gCIIgCIKKKX4G%0AZmGIAYYgCIIgqJhn4RKJGGAIgiAIgooRkzwFQRAEQah34gyGIAiCIAj1TszBEARBEASh3okzGMJT%0ApyZTnXuRSBrajZ2CEjUV+ta1ZuqVxk5BiczsxcZOQYkq3fsD4Guj1xs7BYXlRxY1dgpK1Jqo2M0/%0AYjY3dgbK3v20sTNQWWKAIQiCIAgqRtyLRBAEQRCEelf0DNyvXQwwBEEQBEHFiDMYgiAIgiDUuyIx%0AwBAEQRAEob6JMxiCIAiCINS7Z2EOhup8r08QBEEQhGeGOIMhCIIgCCpGXCIRBEEQBKHeiUmeQqOL%0A+e0Ey1euprBQRker9ngscEZPT7fWMTOc5mJiYsz8ObOVnt++K4pDR46xwt+n9rkdP05gUFBpvVa4%0Au7mip6dXq5iUlBQ+GTeeiLAtGBgYVF9nTExpeYUl5bm7l6+zkpiioiJ8fHw4GRtLUVER48aNw8He%0AHoCjx47h4uKChbm5opz169ezJSyM6P37Fc9lPHhAXl4ep7etqzbXY3+cwX99KIUyGR2fb4vXjC/R%0A021WLk6SJJx9V9OhXRsm2Q9RPB8aFU3k/sMUFBTSxeoFvGZ8iZaW5lNvnxs3bjBv3jzF8kXFxVy/%0Afh1fX1/sbG1ZuXIlhw4fBqBrly44Ozuj3PtK6z0RS8DqNchkMqzav8BCZyf0dHVrFDNzvhtJyXcU%0AcXfuptDn5Z4ELfvnVzG3R+3l0LHfWOHjXWUb/Rvj1/tw9+IVDvj+3GB1PCnmwlWWbz9MobyIjq1N%0A8Rj3IXo6yr+yG3rkFOHH4kAN2pgY4PbJEIz0K9oK1dQVE0NQYCCFhYVYdexYYf+pKu5xH4o9eVLR%0Ah+wdHAA4dvQoLi4umFtYKMpZv349urq6REZEEBISgrq6Oq2bSrh/PAgDvfL7CkDMxRsERsVQKJfT%0AsZUp7h8PKt8ex84Q/tufqKmp0ca4Ja6OAzFqrsvDRwW4h+zjr9QMJEliyKvdmPRO31q3U314Fu6m%0A+szPwfjjjz+YMWNGg9fTrVs3xo4dq3i4u7uTnJyMQ+nO0xAyHjzAZeEi/JYuJmprGJatWxOwYlWt%0AY9YF/8KZs+eUnsvKysJz8VIWL/ODOtx0JyPjAa5u7vgu82HXju20tmzN8sCgWsVERe1m4qRPSU9P%0Ar2GdGbi6ueHr48OunTtpbWnJ8uXLaxwTGRlJUlISWyMjCdm8mc2bN3PhwgUAzp07x/hx4wgPD1c8%0AdHV1+XTSJMX/a9asQUdHh++XLq0+18xsnH1XE+Ayk71rA2hjbobfupBycTeSkpn0nSf7Y2KVnj/w%0A2x9s3rmftYtd2PWTLwUFhWzcvqdR2qd9+/ZK7WJtbc17gwbxtp0dhw4f5mRsLOFhYWzbupX8/HxC%0AQsqvZ8aDTFwWfY//Yg+iwoKxbG1BwKqfahzj5+1BZPAaIoPX4D53Ns2b6+E8+xsAsrKyWbjUj8V+%0AQUgN9KnQvFN7vj0UQm+HDxqk/MpkPMzFZeMu/L6wJ2rhVCyNDQjYfkgp5lLiXTYeiCX4u4lsd5tC%0AW1NDVu46Uvu6MjJwc3XFx9eXnbt2Ydm6dbn+U13c4z4UuXUrm0NCyu1j48aPL7eP3UlOZsWKFaxb%0Av56IyEhaGbZg9d4TlbRHHq6b9+H76VB2uUymtXELlu86ptweSSkEHz5F8MxP2DZ/Em1NDFi55zcA%0AVu45jlnL5mybP4nNs8cS8dufnPvrTkVVNbiiYqnOD1XxzA8wnpYWLVqwadMmxcPd3b3B64z9/RTd%0AunSmXds2ADiMGM7e/dFKB9HqYk79XxwnYn/HfvhHSmVHHzyEsbExs76ZXsfcYunWtSvt2rUtqdfe%0Anr379j2RW+UxaWnpHD56hBVBQRWWX2GdsY/La1d5nVXEHD58mKFDh6KhoYG+vj6DBg5kz969QMnB%0A79Tp04x2dGTCxInExcWVq9/P3583+vXjjTfeqDbXE2fO0e3F9jzXuuTT2ujB77D78G/l3gBDd/3K%0AsHdtGPSmtdLzOw/GMGHEYFrq69GkSRPcvp7Mh3ZvNlr7PHbmzBkOHjzIggULAHjbzo6NGzagqalJ%0Abm4uGQ8e0KJFi3K5nTx1mq6dX6RdG0sARg0fyp7oQ0q51SRGJpPh7LmE776ZirmZKQDRh45iYmzE%0ArOlfVtk+/4bN1HHEro8gLrzqQV59i710k27tWtHOzAgAh7f6sPePC0pt0qVdK6I8p9FcpykFMjlp%0AmQ9poatT+7piY+narZuib9g7OLBv795yfbaquCf70MBBg9i7p6TNzp07x+lTp3AcPZqJEyYo9rGi%0A4mLkcjm5ubkUFxeTXyhDS0O94hwv/0W3tua0MzUsaY83Xmbv/11Sbo+25uxynUxzHW1Fe7RsVtIe%0A342wY+ZHAwD4OzuXQnkRek0b555LxZJU54eq+J+9RHLixAkCAgLQ1tamZcuWeHt7o6uri6urKykp%0AKaSlpWFra8uMGTOYO3cuWlpa3Llzh7S0NJYsWULXrl1rVd/+/fvZvHkzcrkcNTU1VqxYgYGBAR4e%0AHly8eBFjY2Pu3LnD6tWrsbS0rFGZKampioMogJmpCTm5ueTm5ikugVQVk/coj6W+/vwQFEDEth1K%0AZTuMGA7Azqi6HTBTUlIxMzMrU68pOTk55ObmKk6pVhVjamqCv69v7epMTcWszCUMMzOz8nVWEZOS%0Amor5E69dvXYNKBlADh48GDtbW878+SfffvstEeHhivyvX7/OkSNH2B0VVbNc0+9jbmz0T10mRuTk%0APSI375HSZZIF0yYB8PvZi0rL37pzj/uZ2Xw+35u0jAf07taJWZ+NabT2eczXz49p06YpnTbX1NQk%0AdMsWVq5YgYmpKba2thXklo65aZl+alLaT/PyFJdJahKzLWovJsZG2Nn0V8Q5DP8QgB17/rmUVd+2%0ATHcDoJNdvwaroyIpD7IwN/xnwGZmoE9OfgG5+YVKlwU01dU5fPYy7sFRaGlqMHWITa3rSk1Jwbzs%0A/lpB/6kuLjUlpVwfunb1KvDPPmZrZ8efZ87w7bffEh4RQdu2bRk/fjwfDR1K8+bN0VOX2DTrk0ra%0A4yFmBs3/Kb9lc3LyCytuj3PX8Ajdj6aGOl99UPKhQE1NDQ11NeZt3M3Bs1ew7WHFc2aGtW6r+lCk%0AOuOEOvufPIMhSRIuLi6sWLGCX375hVdeeYXVq1dz7949XnrpJdauXUtkZCRbtmxRLNOqVSvWrl3L%0A2LFjCQsLK1dmVlaW0iWSixefeEO4dYuffvqJ0NBQOnTowG+//cahQ4fIzMwkMjISb29v7t27V6v1%0AKJYq/qJ0E/Um1cZISMxxdmXOzG8xMTauVb01IVWam3qtYmpVZ3EN6qwipriC19SblLSlv58fdqVv%0AjL1efpmePXsSG/vPZYuQkBBGjxpF8+bNy5VRkeJKTmOW3XZVkcuLiD1zHj/nbwkPWkzWwxyWr99S%0A5TIN2T4AZ8+eJTMzk/ffe69cnOPo0Rw/fhw7W1tmOznVPLcy5dckZtOWSL6YOLbCuGdRpf2ogjug%0A2r7UiRg/J74c/BZfBm6udNlK66rkk3HZPlBdXEV96HH/8/P3x9bODoCXe/VS7GMnT57k4MGDREdH%0Ac/DQIQb06IDLL3vLlQNUegmswvboacWxJdOZ8l4/pqyKUGqPxeMHc2zJdLLy8vlxn2rd+fe/5H9y%0AgPHgwQP09PQUnz5feeUVrl27RsuWLblw4QKzZs3C29ubwsJCxTKdO3cGwNzcXOn5x568RNKtWzel%0A142MjPjuu++YN28eV65cQS6Xc/PmTV566SUADA0NeeGFF2q1HhZm5qT/fV/xf1p6Ovr6zWmmo1Nt%0AzM2bf3Hnzl18/AOx/3gcEVu3E33gEG5e9TMBztzcnL///vufetPS0NfXV8qtJjG1qtPCovo6q4ix%0AsLAg/YnXzMzMyM7OZs2aNUoHL0mS0NAsmVBZVFTEwUOH+PDDD2ucq4WpMekZmYr/U//OQF9Pl2ZN%0Am9ZoeVMjA+z6vYqebjO0NDUYYtufswnXqlymodrnsejoaIYMHqz0hn/lyhUSLl8GSj4dDhs2jISE%0AhPK5mZuRfv+Jftq8+RP9peqYhCvXkBcV0eflnlW2w7PEwrAF6VkPFf+nZWaj36wpzbS1FM8lpWVw%0A5nqS4v9h/V7i3v0ssvMeVVv+yl1HcHBwwMHBge3btlXYN3SaKU+2tKhkv9Zp1gwLCwv+LjOnqrp9%0ATFNDg2NHj/KWjQ2GRkY0adKEUf17cfpqEhUxN9Tn7+zcf8rPeli+PdIfcOZGsuL/j6y7cy8jm+xH%0A+ZxI+Iu00vZspq3Fe707k5CcWm07NYRn4RLJ/+QAw8DAgJycHNLS0gA4deoUzz33HNu2baN58+b4%0A+voyadIk8vPzFR1eTa38CLimHj58SGBgIP7+/nh5eaGtrY0kSVhZWXH27Fmg5AzIrVu3alWu9Wuv%0Acv7iRRKTbgMQsXU7A958s0YxPXt058CenUSEBBMREoz9iGEMfMcOjwXz67yeSvVaW3P+wgUSE0sO%0ABBGRW7GxeavWMbWu8/x5EhMTS8uLxMbGpsYxNjY27NixA7lcTnZ2NvujoxkwYAC6urpsCQvj0KGS%0AyXMJly9z8eJF+r3+OgDXrl1DX1+f1q1b1zjXfr17cP7yNW7dKTlrFbbnALbWfWq8/Lv9+xId8zv5%0ABYVIksShk6fp3rF9lcs0VPs8FhcXx6t9lWfcX712DTdXVx49Knkzi9q9m1dffbVcbq+/2ofzFxNI%0AvF1y4A/fHsWAN/vVKub//jxH394v/6t99b/Gukt7zt+8Q2JqycArIiaOAT1fVIpJz3rInJ+38iAn%0AD4A9f1ygQ2tTWlbyLYyypn44QDHhctOmTUp9IzIiolz/gfJ9qGzck30oev9+xT4WtmWLYh+7nJDA%0AxYsXeb1fPzp37sxvx4+Tl1eS/8GzV+jxXKuK26PTc5y/dZfEtIyS9vjtLDbdOyjF/J2Vw3cbdina%0AY+/pS3SwMKalrg6/nrnMD/tOIkkShTI5v/55hVet2lbbTg3hWZjk+T8xB+PEiRMMHz5c8b+vry9e%0AXl5Mnz4dNTU1WrRoweLFi7l//z6zZs3i7NmzaGlp0a5dO8Ug5N/Q09OjV69ejBo1SjG5KS0tjeHD%0AhxMTE8Po0aMxNjamadOmaGpW/TXDsowMDfF0XcCsufORyWS0sWzNIndX4i8l4O61mIiQ4EpjGpqR%0AoSEL3d2Z7eSETC7D0tKSRZ6exMdfwmPhQsLDtlQa86/q9PAoKU9WWp6XF/Hx8Xh4eBAeHl5pDJRM%0AaEy+fRt7BwfkMhkjR46kT5+SN/3lAQEsWbqUVatXo6Guzvfff6/42mxSUhKtWlV8wKs015Yt8Jo1%0AhRmefsjkctpYmLPYaSoXr97Axf9Htq/+vsrlHQcPJOthDiOnzaW4uJguHZ5nzudVXxpoyPYBSExK%0AovUT7TBk8GBuJyXx8ZgxqKur0759e9zd3CrIzQDPBXOYOd8NmUxOm9at8HadR3zCFdwWLyMyeE2l%0AMYr6byfTysK8XNnPMiN9XTzHf8isnyKRyYtoY2LAookfEX/rLu6boohw+YLeVu2Y/H5/JvluRKNJ%0AE0xaNidgSu2/3WZoZITHwoU4zZ6t6Btei0q+Bly2D1UVZ+/gwO3kZBzs7ZHJ5Up9KGD5cpYuWcLq%0AVatQ19BQ7GNDP/qIu3fv4jh6NFpaWlhoF+P5yfsVt0dzXRaOeY/Za3ciKyrC0rgli8Z+QHzSPTxC%0AogmfO4FeHdow+V1rPg3cUtIeLfTwnzwMgFnDBuAV9isjFq9HDRjQw4oxNjUf+NcnVTqQS6+RAAAg%0AAElEQVQTUVdqUkN9b0uo1o0bN7h8+TIffPABDx48YPDgwRw5cgQtLa1KlynIzniKGVZN0mic2dWV%0AUlOdE3KaqVcaOwUlMrMXqw96ipo8etDYKSj52uj1xk5BYfmRRdUHPUXFr41o7BSUqMVsbuwUlDR9%0A99MGKTf8/N06L+vQo3YfeBrK/8QZDFVlYWGBj48PGzdupKioiNmzZ1c5uBAEQRD+NzwLZzDEAKMR%0ANWvWjNWrVzd2GoIgCIKKqe23fFSR6pxTFgRBEAThmSHOYAiCIAiCinkWfmhLDDAEQRAEQcWIORiC%0AIAiCINQ7cbt2QRAEQRDqnZjkKQiCIAhCvSuS6v6oi/z8fKZPn87HH3/M5MmTycio+DeXiouL+eyz%0AzwgNDa22TDHAEARBEAQV87TvRRIaGkrHjh0JCQnho48+YtWqVRXGBQQEkJ2dXaMyxQBDEARBEP7H%0AxcXF0b9/fwDefPNNpbtFP7Z//37U1NQUcdURczAEQRAEQcU05CTPiIgINm7cqPSckZERzZs3B0BX%0AV5eHDx8qvX716lV2795NYGAgK1eurFE9YoDxH1OsVf0dEJ8WteKixk5BiVqxvLFTUCgw69TYKShp%0AgmpNGJMf2Fh90FOkSvf/+GaAc2OnoMQnd3j1QU+RRv+PGzuFp6Ih74pqb2+Pvb290nPTpk0jN7fk%0AVve5ubno6+srvb5jxw5SU1MZP348d+7cQVNTk9atW/PmE3fwLksMMARBEARBxTzt26736tWLY8eO%0A0aNHD2JiYujdu7fS63PmzFH8HRQUhLGxcZWDCxBzMARBEARB5RQVS3V+1IWjoyPXrl3D0dGRsLAw%0Apk2bBsD69es5dOhQncoUZzAEQRAEQcU87TMYOjo6BAYGlnt+4sSJ5Z6bPn16jcoUAwxBEARBUDFP%0Ae4DREMQlEkEQBEEQ6p04gyEIgiAIKuZZOIMhBhiCIAiCoGLEAEMQBEEQhHonBhiCSoiJiSEoMJDC%0AwkKsOnbE3d0dPT29GscVFRXh4+ND7MmTFBUVMW7cOOwdHAA4feoU/v7+yOVytLW1mfPdd3Tv3r3y%0AXI4fJzAoiMJCGR2trHB3cy2XS3UxKSkpfDJuPBFhWzAwMADg1OnT+PsHlOTRVJvv5syhe7du1bfN%0A8d9YHrSCQlkhHa2s8HB1qSCfqmNSUlL4ZPxEIraEYmDQEoCsrCwWf7+Mmzdvkl9QwORJkxgy+INq%0A83nS8ZgYgoJKt4lVR9wq2XbVxaWkpDBu7CeEhUco2qzC9oiJKW37knWtqK9UFvO4n5yMjVX0Ewd7%0Ae27cuMG8efMUyxcVF3P9+nV8fX15284OgMLCQqZPn87IkSN55513qm+XK7cJOniGQnkRVuYGuA3t%0Ah15TLaWYLX8kEHHqCmpqYGnYHNcPX8dQT4esvAK8d8dy5V4GOloafPiyFY6vda62zpqKuXCV5dsP%0AUygvomNrUzzGfYiejrZSTOiRU4QfiwM1aGNigNsnQzDS1623HGpq/Hof7l68wgHfn+u97OPHY1gR%0AFISssJAOVla4ulXSd6uJS0lJYcK4sYSGhSv67s0bN/Dy8uRRXh5qampM+/obXn/9dcUyDdGP4fFx%0A5p/j3Xdz5tC9e3fWrltH9P79irIzHjwgLy+PkydO1GublvUsDDDEJM//uIyMDNxcXfHx9WXnrl1Y%0Atm7N8uXLaxUXGRlJUlISkVu3sjkkhM2bN3PhwgVkMhlz5szB1dWV8IgIJk+ezALnyn9lMCPjAa5u%0A7vgu82HXju20tmzN8sCgWsVERe1m4qRPSU9PVzwnk8mY891cXF1diAgPY/Jnn+G8wKX6tnnwABd3%0AD/x8vidq+zYsW7cmIGhFrWJ27d7NhE8nk1YmH4AFbu6YmZoSHhrCT6tXsWSZDympqdXmpNwWGbi5%0AubLMx5cdO3dhadmawMq2XRVxUVFRTJo4UanNKqvP1c0NXx8fdu3cSWtLy3J9paqYx/1ka2QkIZs3%0AK/pJ+/btCQ8PVzysra15b9AgxeDi3LlzfDJ2LH+ePVuzdsnNx23HCZaNHsCOb4ZjadCcwANxSjGX%0A7v5N8ImLbJj8PpHTPqKtoT6rDv8JgM/+U+hoabJ1+kcET/6AE9eSiblyu0Z1V5vbw1xcNu7C7wt7%0AohZOxdLYgIDtyr8RcCnxLhsPxBL83US2u02hrakhK3cdqZf6a8q8U3u+PRRCb4faD3pr4kFGBh5u%0Abixb5sO2HTuxtLQkKLB8360ubndUFJ9NKt93lyz2ZujQjwgNC8fV3YO5381BLi/5pd6G6sdlj3cR%0A4eFMnjwZ5wULAPh00iRF/16zZg06Ojp8v3Rpvbbpk57272A0hHodYPzxxx/07t2be/fuKZ7z8fFh%0A27Zt9VL+kiVLGDt2LIMGDcLGxoaxY8fy9ddf10vZdXX69GkuX74MoPhhkqcpNjaWrt260a5dOwDs%0AHRzYt3cv0hO/Y19V3OHDhxk6dCgaGhro6+szcNAg9u7Zg6amJr8eOECnzp2RJInk5GRatGxZeS6/%0Ax9Kta1fatWsLgIO9PXv37VPKpaqYtLR0Dh89woog5UGJpqYmB6L307lTp9I87tCyRYsatM3vdOva%0AhXZtH9c1snw+VcSkpadz5MgxVgYpH7yysrL4/Y9TfPn55wCYm5mxOXgDLfSrz6ms32Nj6dq1zDax%0Ad2DfvvLbrqq4tLQ0jh45TNCKFeXKL98ej9u+Xem6VrB9qoh5sp/8P3vnHRXV1fXhh95R6ShqFBsS%0AU4wajRqxJFGj0SQCGmuiGDUxViyoVDUWFHuMxi4IA/beUDH2kgREY1dEP0BAadKGme8PZGSAwRJw%0A5tXzrMVazMy+5/zuPufeu++pnb/4gt179ijlcfHiRQ4dOsTUpzdmgJBNm/j5p59eqMUJ4PSN+zhX%0At6K2ZeFSxa7NG7I3+paSzsbVrdg+6lvMDPXJzZeSlPGEKk9bEa48SKHb+3XR0dZGT1eHtg0cOBR7%0A54Xyfh6nLt/i3drVqW1rCYBbu2bsOROjrK12dXYG/IyZkWGhtscZVDExqpD8XxSXnwZwak04FyS7%0AKyX9U6dP0djZmVpP60kvV1f2lqhLz7N7mJTE0aNHWLS4dN0tkMkUO3Y+ycrCQP9Z61Vl1WM9PT0O%0AHjhQ7D4TX+Z9Zn5QEG1at6ZNmzb/xYXPRSqTv/KfplDhXST6+vpMnjyZNWvWoKWlVaFpT5o0CYAt%0AW7Zw69Ytxo8fX6HpvwqbN2+ma9euNGrUiCUvcJOvaBITErCztVV8trW1JTMzk6ysLKUmw/LsEhMS%0AsLOzU/rt+rVrQOHDPSUlhd7u7jx+/JjZc+ao1JKQkIht8TxsbEppKc/GxsaaoHnzyky7SId7n+94%0A/Pgxc2bPeq5vEhITlc/ZxobMzCxlPeXY2FhbEzRvbql04+7dw8rKig3BG/nzxEny8vIZOKAf7zy9%0Amb0oCYkJ2No9y9tGRdmVZ2djY8O8+UEvmF8itiXKuVT5lGOTkJhYqp5cu35dKY958+fz888/K+mf%0APauwrEpurqRSZ1oWtlWe7bljY25CZm4+Wbn5St0kejraHLlyF//tJ9HT0WH4Dx8C8K6DNbv+ucX7%0AtWzJlxZw+PJddLUr5l0q4VEadhbPHjq21czJzMklKydPqZtET0eHyL//xXf9TvT1dPmpu0uF5P+i%0AhI70AaBRx9aVkn5iQiJ2ts/qgo2NLVll3ndU21nb2BA4b36Z6U+aNJkffxxKSPBGUlNT+XXWbHR1%0ACx9XlVmPFfeZ3r2f3meUWylu3LjBkSNH2LVz50v77G2kwgOMli1bIpPJCA4Opl+/forv4+PjGTt2%0ALBKJBAA3Nzfmz5/P1q1buXv3Lo8ePeLx48f07duXAwcOcPv2bWbPns0HH3zw3DzPnDlDYGAgenp6%0AuLm5YWhoSHBwMFKpFC0tLZYsWcL169dZuXIlenp6xMfH07VrV4YPH86BAwdYuXIlurq62NjYEBQU%0ARFJSEr6+vuTm5vLw4UNGjx5Np06dOHLkCEuWLEEul+Ps7Iy7uzvHjx8nNjaWevXq4erqyokTJ7h8%0A+TIBAQHo6OhgYGBAQEAAMpmMcePGYWdnx71792jSpAl+fn5cuHCB2bMLLx4jIyMWLlxYZj+mKmQq%0AdtzTKXFDLc9OJpOV+l5bR0fxv6WlJQcPHeLKlSsM9fDAsW5dar/zTqlj5PLS6ZRM60VsVGFpacmh%0AA/u5cuUKHj8Oo27duuU+1Ms6r5J5vYhNSaRSKffv38fExJT1a1YTF3ePQUOGULtmLRo3fvG+frmK%0ANw0dHe1Xsnt+fi9QPuXYlOWr4vXs77//5vHjx3Tt0uWldJXSqeIFTEe79AtLe6fatHeqzZbz1/hp%0A/QF2jPqWcV80Y/7+8/T5bQdWpkZ87Fid6Lik/6SpCJmKstAuQ1uHDxrR4YNGRBy/yLBFwewOGFmm%0A3f8iqq5jnRLXzYvaFSc3N5dJkybi6+fPp59+Skx0NKNHjaKxszN2dnaVXo8tLS05dPBg4X1m6FDq%0AOjoq7jMhISH0dndX7DpamWhSV8erUiljMHx9fVm7di137959IXtDQ0NWrVrFF198wbFjx1i+fDlD%0Ahw5l9+4Xb97Lzc0lJCSEnj17cufOHVasWMGmTZuoV68ef/75JwAPHjxg8eLFhIWF8ccffwCwa9cu%0ABg8ezKZNm2jfvj2ZmZncunWL77//njVr1uDv768IVgICAlixYgVbtmyhVq1aWFhY0LZtWzw9Pale%0AvbpCy9SpU/H29mbjxo306dOHWU/f4O7cucOMGTMIDw8nKiqKhw8fcujQIbp06aKwLWoWLI9lS5fi%0A5uaGm5sbW7dsITk5WfFbUlIS5ubmGBkr77pqb2en0s7e3p7kYn2gSUlJ2NrakpGRQWSxNeidnJxo%0A0LAh12/cKFOXnYo8jI2MXsqmJBkZGRyOjFTS0bBBA25cL1tH8XN+qJTXw1J5vYhNSaytrQHo0b0b%0AALVq1eTDDz4gJja2XD0Ay5Ytxd3NDXc3N7ZuVVF2RsplZ2evouyMXm5nXTt7++eXTzk29vb2JXyV%0ApNQatX//frp364b2f2wtsKtqQnJm9rN8Mp5gbqSPkb6e4ru4lHT+uvtszEuPpvX4v8dZpOfkkpmb%0Az+jPmxHxc0+WD/oCbS0taloq7wz5qthbVOFh2rNtrJMep2NubIixwbOWlbikVC7eiFN8/rr1B/xf%0AShrpT7L5X+a3Zcvo4+5GH3c3tm3dqlRPHirqpPJ1Y2dn/0J2xbl54wY52TmKjbSavPcejo51uRQT%0AU5hmJdVj1feZwtaNgoICDh0+zFdfffViDvuPiDEYKqhWrRpeXl5MnDhR5RuiUp9l48YAmJmZUa9e%0APQCqVKlCbm7uC+dZp04dxf+WlpZMnDiRyZMnc/XqVcXgoAYNGqCrq4uxsTGGhoYATJ48mdOnT9Ov%0AXz8uXryItrY21tbWhIWF4enpSWhoKFKplEePHmFubo6lZWHfq4eHh1JQUZykpCScnArfZJs3b871%0ApxW0Vq1amJqaoqOjg7W1Nbm5uQwbNoykpCQGDhzIvn37FM2A5THip58UA442bNhAdHS0IpiLCA/H%0AxcWl1DGtWrVSaefi4sK2bduQSqWkp6ezf98+2rdvj46ODj4+Pvz1V+HguRs3bnDn9m2Vs0hatWpF%0AdEwMd+8W3lzDIzbj4tLupW1KoqOjg4+vn2KQ4I2bN7l95w5NmpTfp9+qVUuiYy5xN+5pXps3075d%0AST3PtymJQ40aODVqxI5duwBISUnhn3+icX6B1osRI34iTCIhTCJh/YYNxBQvkwjVZfcids+jZB0I%0Aj4golU55NiXryb79+2nfvr3i2AsXLtDi449fWlcpnY7Vibn3kLsphcF2xLmruDSqpWSTnJHNpPBj%0APMrKAWBP9C0cbapS1diQiHNX+e3pgM+UzGy2XrhGl/fq/mddAK0aOxJ96z53E1MACI+6QPv3GyrZ%0APEzLYMLKzTzKfALA7jMx1KthQ1XTlwsINY3hI0awKUzCpjAJa9dvICYmmjhFnYygXRl1smWrVi9k%0AV5yatWqSmZnBP0+v93v37nH79m0aNmoEVF49Lut+V3ifKbzfXb9+HXNzc2rUqPFyjntFCuTyV/7T%0AFCptmmqHDh04ePAgW7duxdPTEwMDA1JSUigoKCArK4v4+HiFbUWM1Sh6a8rIyGDRokUcPXoUKNyo%0ApSiYKSufsLAwRo4ciaWlJd7e3hw8eJDDhw/j6upKu3bt2Lx5M1u3bsXS0pL09HQeP35M1apVmT59%0AOl999RVaWlqlBjbZ2Njw77//0qhRI86dO8c7T7sTysp/x44dfP3110ycOJHff/8diUTyUoNFLSwt%0A8fP3x3P8ePLz83FwcGD6jBkAxMbG4ufnh0QiKdfO1c2Ne/HxuLm6ki+V0qtXL5o1awZA0IIFzJ07%0AF6lUir6eHr/++qvSW2txLC0s8Pf1ZbynJ/nSwjxmBAQQG3sZP39/JGGhKm3Kw9jYmAXz5zN3biBS%0AqRQ9fX1+nTlDpY7iegJ8vRnnOZH8/HxqOjgwI8CP2MuX8fWfTnhoiEqb57FgXiAzZs0mPGILMpmM%0AHz2G8K6z83OPK46FhSW+fv54eo5H+rRMAqY/Kzt/Pz/CJJJy7V4GSwsL/P38Cn3/NJ0Z06cr1RNV%0ANlA4UC7+3j1c3dyQ5ucr1ROAu3Fx1FARdL+UX0yN8P26DZ6hR5AWyHCwMCPgm7bE3k/Gf/sJwkb0%0AoOk7tgz+9D081uxDR1sLazNjgr7rAMAPn77H1M1R9FqyDbkcfmz/Ac41rP6zLgBLcxMCBn7FuBUR%0A5EsLqGldjRnf9yT2zgN8N+wkfNqPfFS/Nh5d2/LDvHXoamtjXdWMBcPdKiR/TcHCwgIfXz8mFLuO%0A/QMK68nl2FgC/P3YFCYp104VZmbmBM4PInDuHHLz8tDV1cVr6lRq1qwJVG49XhAUpLjf6enrK93v%0A4uLiVL5UVgaa1BLxqmjJSz4d/wNnzpwhNDSUoKDCQWeZmZl0796dkSNH8s033+Dt7U1MTAw1a9Yk%0AKSmJwMBAtm7dipWVFX369GHTpk0kJyczcuRIDh06RFRUFP7+/qXyKTnIs3i+crmc0aNHc//+fcUo%0A4Q8//JCmTZsqaWvdujUnTpwgMjKSZcuWYWJigrGxMTNnzuTEiRP89ttvVK1aFTs7O/799192797N%0AsWPHWLZsGdra2jRu3JipU6cSFhZGcHAwCxYsYMCAAYoxGDNmzEAul6Ojo8PMmTPR0tIqcwxKSkoK%0AM2bMwMjICG1tbfz9/RUXUllk5+RUVHH9Z7RkBeqWoISWiv5edVCgrfd8o9eINpp1s5LtWKBuCUro%0A2NZ6vtFrYlR71VPB1UFg1hV1S1BCV8OGsRiW093zXxgc+tcrH7uq94cVqOTVqdAAQ1D5iABDNSLA%0AUI0IMMpHBBiqEQFG+VRWgDEo5OIrH7v2u6YVqOTVEQttCQQCgUAgqHDEUuECgUAgEGgYBSomSPwv%0AIQIMgUAgEAg0jDdhkKcIMAQCgUAg0DBEgCEQCAQCgaDC0aQ9RV4VEWAIBAKBQKBhiBYMgUAgEAgE%0AFc6bEGCIaaoCgUAgEAgqHNGCIRAIBAKBhvEmtGCIAEMgEAgEAg1DBBgCgUAgEAgqHBFgCF47mrT/%0Ah1xbR90SlJGLIUWqkFfAjsUViaz7aHVLUEJLW3P8E5j1jbolKDHexEndEpSY/+RfdUt4LchFgCEQ%0ACAQCgaCikYkAQyAQCAQCQUXzJmx0LtqUBQKBQCAQVDiiBUMgEAgEAg1DjMEQCAQCgUBQ4YgxGAKB%0AQCAQCCocuUzdCv47IsAQCAQCgUDDeBMGeYoAQyAQCAQCDUN0kQjUTtTx4yxavJi8vHwa1K+Pr483%0ApqamL2WTkJBAvwEDCQ8LpVq1agCkpaUxa/Ycbt66RW5uLkMG/0D3bt1U64iKYvGiReTl5VG/QQN8%0AfX1L6SjPrqCggMDAQE6dPElBQQEDBgzA1c0NgHNnzxIUFIRUKsXAwIAJEyfSpEkT5HI5S5cuZf/+%0A/RgZGfHBe+/RsmVLflu+nLy8vMJzLUNHVFTUU38o2xRpOHnqlEKDm6ur0rFbt20jMjKSxYsWKb4b%0AO24c165dw9jICIDmzZszznOC4vfjUVEsXvz0nOs3wEeFb1TZFRQUMC8wkFOnCn3Tf8AAXF2f+ubc%0AWebPm0dBQQFVqlRhvOcEGjZsCMCFCxdYuCCIlJQUUlNTsbS0xKlx4wotm7t37+Lr40NaWhpGRkZM%0AnzGDOnXqlCqb999/n/Hjx2NgYEBBQQG///47x44dIyc7m9Zt2jB23Hi0SiwGdvx4FEsWLyY/L496%0A9evj7aPCbyrscnJymD3rV2JjY5HLZLzbpAkTJ03G0NCwVBrP80FF+OrY0aNMmzYNO3t7RTpr1qzB%0AxMSEiPBwQkJC0NHRoXr1Gkzz8VFcixXljyISEhIYNKA/m8Ikijxu3bzJ9OkBZD95gpaWFj//MopP%0APvlEpZ9ehYFrAnlw6SoH562s0HSL+K/XWREJCQkM6N+PMEl4qTLYtm0rRyIjWbhocaWcQ3HehEGe%0AYprq/zCpqY/w9vFl3txAdmzbSg2HGqUq/vNsdu7cxfc/DObhw4dKx03z9sHG1gZJ6CZWLP+N2XPm%0AkpiYqEJHKj7e3gTOm8f2HTtwqFGDhQsXvpRdREQEcXFxRGzeTHBICMHBwcTExJCfn8+ECRPw9vZG%0AEh6Oh4cHU6dMAWD79u1ERUURHByMRCLBxNQUrylTmBcYyI7t26nh4FBKR2pqKt4+PmXaFGnYHBFB%0ASHCwQgMUBlwB06cza9asUk2X0dHRrF61ColEgkQiwdPTU/mcfbyZGziPbdt34OBQg0WqfKPCbvNT%0AXeERm9kYHEJIcDCXYmLIyMhg3NixjB4zFkl4BF5TpjJxgid5eXkkJiYybuwYfvp5JNnZ2fTr14/a%0AtWtXaNkAeE2ejKubG1u2bmX4iBGMGzsWuVxeqmysraxYumQJACHBwVw4f57Va9YSKgknOjqaA/v3%0AK+l5lJqKn48Pc+cGsmXbdhwcHFi8qLTu8uxWr/qDAmkBoWESQiXh5Obksmb16lJpvIgPKsJX//zz%0ADwMGDlTUE4lEgomJCffj41myZAmr16whPCIC++r2/L78twr3B8CunTsZ8sP3pa73Wb/OpEePnmwK%0Ak+Dt68ekiROQSqUqffUy2DVyZPThED5y+7JC0iuLirjOAHbu3MkP35f2T1paGtOnBzC7jOtfoJrX%0AGmCcOXOGVq1a0b9/f/r374+bmxsbNmx46XQ2bdrE4sUvF0E+fvyYnTt3PtfO19eXnj17vrSm/8qD%0ABw+IjIx8qWNOnT7Fu87O1K5dCwA3V1f27N2rdAGUZ5OU9JDIo0dYUsKXaWlpnD5zhmFDhwJga2vL%0Axg3rMTc3L1vHqVM4v/sutWvXBsDVzY29e/aUuhDLs4uMjKRHjx7o6upibm7OF507s2f3bvT09Dhw%0A8CCNnJyQy+XEx8dTpWpVAK5cvkz79u0VuszNzJDL5Yr0y/THqSJ/lLYpqaHzF1+we88eAPYfOIC1%0AlRXjxo5VOqf4+/fJyspi+owZ9HJ1ZZq3N2lpaYrfT586hbNzsXN2dWPv3tK+Kc+ulG++6MzuPbuJ%0Ai4vD1NSMjz/+GIA6depgYmJK9D//cOjgQVq3bk1KcjLOzu/iMXQonhMmVGjZJCYmcufOHTp37gxA%0AmzZtyM7J4d9//y1VNh06duTgoUMA7Ny1i8FDPDA0NERfX5+5gYG0aNFCWc/pUzR2dqbWUz29XF3Z%0AW6Isn2f3YdOmDPbwQFtbGx0dHRo2asj//d8DVFGZ9RgKA4xzZ8/Sp3dvvh80iAsXLgBQIJMhlUrJ%0AyspCJpORk5ODvr5BhfvjYVISR48eYdHiJaXOvUAmIz09HYAnWVkY6Our9NPL4vLTAE6tCeeCZHeF%0ApVmSirjOkpKSOHokksVLSvvnwIH9WFtZM2bsuEo7h5LIZfJX/tMUXnsLRsuWLdmwYQMbNmxg48aN%0ArFmzRlGxK5OrV68+9wGenZ3NhQsXcHR05MyZM5WuqTinT5/m4sWLL3VMQkIitra2is+2NjZkZmaS%0AlZX1QjY2NtYEzZuHo2NdpXTj7t3DysqKDRuDGTjoe/p815crV/7F6GkXQEkSExKwK56HrW0pHc+z%0AS0xIwM7OTum3ohYTPT09UlJS+PyzzwgKCmLQoEEANGnShGNHj/Lo0SNkMhnH//yT/Pz8cnUkJCZi%0AWyKfIpuExESVGtxcXRk2bBgGBso3/tTUVD7++GOmTZ1KWGgoxsbG+Pj4FMsvAVu7Z+dso8I35dkl%0AJiYoabaxtSUpMZHatWuTnf2EUydPAhB76RK3bt3kYXIyd+/excjIiLVr1xATE83ECRPQ09Or0LJJ%0ATEzE2toabe1ntxFbGxsSExNLlc2unTtJfvpWGHf3Lrdu3WLYj0Nxd3MlIjwc8ypVSuhJxM622Dnb%0A2JJVpm7Vdq1afaJ4kPzfgweEBIfQ6bPPUUVl1+MqVarg7u7OptBQfvnlF8aOGUNiYiK1atVi4MCB%0A9OzRg04dO3LxwgV+GDy4wv1hbWND4Lz51HV0LHXukyZNZs2a1XT54nOGD/uRyV5T0NWtmB700JE+%0AnNm4tULSUkVFXGc2NjbMmx+EYxn+cXV148dhwzAscf1XJjK5/JX/NAW1jsHIzMxEW1ubQYMGUbNm%0ATdLS0lixYgVeXl7Ex8dTUFDA999/T9euXTl//jwzZ87E3NwcHR0dPvjgA+Lj4xk7diwSiQQANzc3%0A5s+fj7GxMRMnTiQjIwO5XM7s2bNZvnw5//77L2FhYVSrVo2VK1eiq6uLjY0NQUFBaGtrs3fvXlq1%0AasWnn35KcHCw4s2we/fuNGvWjKtXr1K3bl0sLS05f/48+vr6rFixguzsbDw9PcnMzKSgoIBRo0bR%0AqlUrOnTowN69ezEwMCAwMJC6detSo0YNVq5ciZ6eHvHx8XTt2pWhQ4eyYsUKcpkYQLAAACAASURB%0AVHJy+PDDD+nYseML+U+uYh6Tto7OS9mURCqVcv/+fUxMTFi3dg1xcXF8P3gItWvVpHHjxqXsVVVo%0AHW3tF7aTyUrrLK7R0tKSg4cOceXKFYZ6eOBYty7duncnMSmJoR4eGBkZYW1lpfSwKysdeRn5FNmU%0ApaHkOZTkvSZNWBAUpPg8fNgwOnbqRH5+Pnp6eirfJnR0lNMtz65M32jrYGpqSlDQApYsWULQgiCa%0ANm1K8+bN0dPTQyqVEhV1jM5dupCTk4Nj3bqMHTuWkJCQMs/rVcqmrO8LtWmXKptvvv0WPT09oLB+%0AxcREs2jxEvLz8xkz6hfCQjfxXd9+z/yhot7qlKi3L2J35fJlxo0bi3tvdz799NMy7aHy6/H8YvXk%0Aw6ZNef/99zl16hQ2NjYcOnSI/fv3U7VaNYKCgvD18WbBwmfjfCrSHyXJzc1l0qSJ+Pr58+mnnxIT%0AHc3oUaNo7OysFCxpMhVxnWkamtQS8aq89gDj9OnT9O/fHy0tLfT09Jg2bRp//PEH3bp147PPPmPj%0Axo1YWFgQGBhIZmYm33zzDS1btsTPz49FixZRp04dpTfEsli2bBkdOnSgT58+XLx4kejoaIYNG0Zo%0AaCju7u788ssvDB48mM6dO7Nt2zYyMzMxNzcnPDwcf39/HB0d8fX1JTGx8O0/KyuLbt264ePjQ+fO%0AnZk8eTJjxoyhX79+3Lhxgx07dvDJJ58wcOBAEhMT6dOnD4cPH1ap78GDB+zYsYO8vDzatm3L8OHD%0AGTp0KLdu3Xrh4ALAzs6OmJhLis9JSUmYm5srBhu+qE1JrK2tAejxVXcAatWqxYcffMClS7GKAGPp%0Ast84GhUFQFZmJvXr1y+Vh5GxsVK69nZ2XHraH13Szt7eXvGGW/Sbra0tGRkZnDt7lg5P/eLk5ESD%0Ahg25fuMGVatVo0uXLgx++ra3/LffOHP2bPn+sLcn5lLZ/rC3t+dhcnIpDeVx8eJF0tPTcXFxAQqn%0AlslkMvr1/Q7QIisrk3pl+cZI2Td29nbEXCrDN0bG2Nnbk5z8zDcPn+qSyWQYGRvzx6pVit+++bon%0Ap06dJCrqGLm5uRw9coR69eszftw45syZw7179yqsbOzt7UlOSUEulysGaBb9lpaWplQ2MdHR1KxZ%0AEyisX1980Rl9fX309fXp9NlnXLx4kbS0ZUQdOwpAVlYW9eo989tDhT+U662dnT2XitXvknb79+1j%0A1q8zmTBpEl26dC1Vfr8te5qnllal1uP09HQkEgmDBw9W+Eoul6Onq8uxo0dp5+KChaUlAG7u7rj1%0A6vVMWwX6oyxu3rhBTnaOIvhq8t57ODrW5VJMjEYHGMuWLeXY0WMAFXKdaRpvQoChti6S9evXs2rV%0AKtq1awcU9h8D3Lx5k+bNmwNgamqKo6Mj9+7dIzk5WWHTtGnTMtMu6m+7ffs2H374ocL2q6++UrKb%0APHkyp0+fpl+/fly8eBFtbW1u3rzJ9evXmTVrFh4eHmhpabFp0ybFMc7OzgCYm5srmtDMzc3Jzc1V%0A0mxra4upqSkpKSllagNo0KABurq6GBsblzui/Xm0atWK6JgY7t6NAyA8YjMuLu1e2qYkDjVq4OTU%0AiB07dwGQkpLC3//8Q2PnZ60XP40YrhiotmHDBqKjo7l79y4AEeHhigduKS0q7FxcXNi2bRtSqZT0%0A9HT279tH+/bt0dHRwcfHh7/++guAGzducOf2bZo0aUJsbCxjx4whPz+/8K04OhqZTKZIPzwiopSO%0AkhqK25TUsG//ftq3b1+ur548ecKs2bMV4y7WrltH5y++IEwSTphEwvoNG4gpfs4Rqn2jys7FxYXt%0AT3VlpKezf/8+XNq3R0tLi5E//0RsbCwABw8cQFdXl6lTp7F06TL09fWZPWcOMdHRhIWF4ejoyM4d%0AOyqsbGxtbanp4MD+ffsAOHniBNra2tSvX79U2axavZquXxYO8uvUqRN79uxGJpORn5/P8ajjNHZ2%0AZviIEWwKk7ApTMLa9RuIiYkmTuGPCNqVobtlq1Yq7Q4dPMjcObNZuuy3MoMLQJFnZddjExMTwkJD%0AFS8e/165wqVLl/ikdWucnJz48/hxnjx5AsDhQ4dp0uS9CveHKmrWqklmZgb//P03APfu3eP27ds0%0AbNSo3OPUzYgRPxEmkVTYdaZpyGTyV/7TFDRmmmpRVO/o6Mj58+f57LPPyMzM5Nq1azg4OGBra8vN%0AmzdxdHQkJiaGKlWqYGBgQEpKCgUFBWRlZREfH69IIyYmhkaNGnHu3DmOHj2Ki4uLovkyLCyMkSNH%0AYmlpibe3NwcPHuTq1auMGTOGvn37AoWtDO7u7owYMUJJX1kUaW7cuDGJiYmkp6dTtWpV9PX1SUpK%0AwsHBgX///VcRmJSVlraK5tXysLSwwN/Xl/GenuRL83FwcGBGQACxsZfx8/dHEhaq0uZ5BM2bx8xZ%0AswiPiEAul/HjUA/efRpklcTC0hI/f388x48nP78wj+kzZgAQGxuLn58fEomkXDtXNzfuxcfj5upK%0AvlRKr169aNasWaGWBQuYO3cuUqkUfT09fv31V2xtbbG1teXChQu4uboik8lo37497u7uhef6NP0Z%0A06crabC0sMDfz6+UDRSOs4i/dw9XNzek+flKGlTRpk0b+vTpw8BBg5DJZNSvXx8fb+9nvrGwxNfP%0AH0/P8Uif5hcw/Zlv/P38CJNIyrVzdXUj/l487m6u5Ocr+2bmr7MI8PcjPz8fK2tr5gctQEtLi4aN%0AGuE1ZQr+fv7o6+uzfPlyLC0tuX79eoWWzazZs/H392flypUYGBgwNzAQbW1tPvnkk1Jl069fYRfI%0ATz//TFDQAtx69aKgQMrHLVvy3Xd9leuUhQU+vn5MKFZv/QMKy+lybCwB/n5sCpOUa7dk8SLkcgjw%0A91Ok+/4HHzBpspda6vGChQuZPWsWvy1bho6uLnPmzKFatWr06NmTBw8e0Kd3b/T19bGzt8fX37/C%0A/aEKMzNzAucHETh3Drl5eejq6uI1daqixel/gYq4zjSNN2G2ipb8NZ7FmTNnCA0NJahYXyRA//79%0A8fX1xdHRkby8PKZNm0ZcXBy5ubn079+fr7/+mujoaPz8/DA1NcXExAQnJydGjhyJt7c3MTEx1KxZ%0Ak6SkJAIDAzE2NsbLy0sxwGfmzJno6+szaNAg3N3dqVWrFsuWLcPExARjY2MCAgLo2bMnO3bswMLC%0AQqHLw8ODr776iqCgIMVYiqJxHg4ODowYMYKhQ4fyzjvv4OXlRVpaGjk5OYwaNYpPP/2UiIgIVq1a%0ARY0aNTAzM6Nt27bUqFFDyQetW7fmxIkTXL58mTFjxvDLL7/w5Zeqp3PlPMlS+dvrRq6tul9XHWhp%0A0AUpQ3VAqg7KiY/VQoEGvWUB6GhrjoM0zTfjTZzULUGJ+U/+VbcEJYyNXr0Vujw+8Nrzysf+PbPs%0AFrvXzWsNMAT/HRFgqEYEGKoRAUb5iABDNSLAKJ/KCjDen/TqAcY/szQjwNCYLhKBQCAQCASFaNJY%0AildFBBgCgUAgEGgYb8IsEhFgCAQCgUCgYYgAQyAQCAQCQYWjSStyvioiwBAIBAKBQMN4E1owNG99%0AVIFAIBAIBP/ziBYMgUAgEAg0jDehBUMEGAKBQCAQaBive5pqTk4Onp6epKSkYGJiwuzZs5UWngRY%0AvXo1u3btQktLi2HDhvHZZ5+Vm6boIhEIBAKBQMOQy+Wv/PcqbNq0iQYNGhASEkLPnj1ZtmyZ0u/p%0A6emsX7+e0NBQVq9ezcyZM5+bpggwBAKBQCDQMOQy+Sv/vQoXLlygbdu2AHz66aecOnVK6XcjIyOq%0AV69OdnY22dnZ5e7PVYToIhEIBAKBQMOozC6S8PBw1q1bp/SdpaUlZmZmAJiYmJCRkVHqOHt7e778%0A8ksKCgr48ccfn5uPCDD+x9Ck/T+0ZAXqlqCERvlG3QJKoEn7tADoapqDooLVrUCBbtvv1C1BCU3b%0A+2OssWZtI79cfqdS0pVX4v3V1dUVV1dXpe9+/vlnxQahWVlZmJubK/0eFRVFUlIShw8fBmDw4ME0%0AbdqU9957T2U+ootEIBAIBIK3nKZNm3Ls2DGgMJj46KOPlH6vUqUKhoaG6OvrY2BggJmZGenp6eWm%0AKVowBAKBQCDQMCqzBaMs+vTpw8SJE+nTpw96enrMmzcPgDVr1lCrVi06duzIyZMncXNzQ1tbm6ZN%0Am9K6dety0xTbtf+PkZ2To24JCkQXyf8OmtZFonEcD1G3gmdoWBeJTMM6/N6WLpJagza88rFxa/tX%0AoJJXR7RgCAQCgUCgYcgLNOsF7lUQAYZAIBAIBBrG6+4iqQxEgCEQCAQCgYYhAgyBQCAQCAQVzpsQ%0AYIhpqgKBQCAQCCoc0YLxP0pUVBSLFy0iLy+P+g0a4Ovri6mp6QvbFRQUEBgYyKmTJykoKGDAgAG4%0AurkBcO7sWYKCgpBKpRgYGDBh4kSaNGmCXC5n6dKlRD5daMW5cWOmeE3m3PnzLFq8mLy8fBrUr4+v%0Aj3cpLVHHj5dpU1BQQOC8+Zw8dYqCAikD+g/AzbUXAGfPnWPe/CAKCqRUqVKVCePH07BhAwDGjhvP%0AtevXMDYyBqB582aMnzCx0n2TlpbGrFmzuHXzJrm5uQwZMoRu3buzetUq9u3fr0j7UWoqT5484cTJ%0Ak2opK4CI8HBCQkLQ0dGhRo0a+Pr4UK1aNaX8Csskr7BMytClyqZIU2G5FWpyK7FwT/z9+/Tp04fl%0Av/2Gs7Nzmef7OvMvqr+HIyOBp/V3yhSMjIxKaSul9dJNFu2MIk8qpUF1G3y/64ypkYGSzaZjF5H8%0A+RdaWlrUtKqKd58vsDQzISM7F9+QvdxOTEUul9O9xbv88NnHz8+zkvxz9tw5pTozccIEmjRpwqrV%0Aq9m/b58i7dRHj3jy5Al/njhZpr7jUVEsXvy0vtZvgI+Kev08u4SEBAb070eYJFypfgJs27aVI5GR%0ALFy0+Ln+elkGrgnkwaWrHJy3ssLTrghEC4ZALaSmpuLj7U3gvHls37EDhxo1WLhw4UvZRUREEBcX%0AR8TmzQSHhBAcHExMTAz5+flMmDABb29vJOHheHh4MHXKFAAiDx/m1MmThEkkbN6yhZycHP5YtRpv%0AH1/mzQ1kx7at1HCoUepmkJr6SKVNxObNxMXFsTlcQsjGjQSHhBBz6RIZGRmMHTeesaNHESGRMNVr%0AMp4TJ5KXlwdAdHQ0q/9YhSQsFElYKJ7jx1e6bwC8p03D1saGMImE31esYPbs2SQmJvLD4MFIJBIk%0AEgl//PEHRkZGzJ4zR21ldT8+niVLlrB6zRrCIyKobm/Pb7/9ppSft48P8wID2bF9OzUcHErpKs+m%0ASNPmiAhCgoOVfASQm5vLFC8v8vPzy6jB6sn/cGQkJ0+dQhIWxpbNm8nJySEk5PnTU1MznuAdvJd5%0Ag3uwY5oHNayqsHDHMSWby3EJrI88y/qx/dji9QO1rKuxdPefACzdfRzbqmZs8fqB4PH9Cf/zL/65%0Afb/8PCvJP8XrTLhEgoeHB1OmTgVg8A8/lKrDc2bPVqnPx8ebuYHz2LZ9Bw4ONVikql6XY7dz505+%0A+P57Hj58qHRcWloa06cHMHvWrFfevEsVdo0cGX04hI/cvqzQdCsauazglf80hTcmwJg1axb9+/en%0Ac+fOuLi40L9/f3755ZeXTmfYsGEvtMZ6RXP16lXOnTv3QranTp3C+d13qV27NgCubm7s3bOn1IVY%0Anl1kZCQ9evRAV1cXc3NzvujcmT27d6Onp8eBgwdp5OSEXC4nPj6eKlWrAtCxUyfWrluHnp4eWVlZ%0ApKamkpyczLvOztSuXQsAN1dX9uzdq6Tl1OlTKm0iI4/Qo8dXCh2dv/ic3bv3EBd3DzNTUz7+uPBN%0Ar06dOpiamPBPdDTx9++T9eQJ02fMoJebG9N8fEhLS6t036SlpXH69Gl+HDYMAFtbWzZu3FhqSd2g%0A+fNp3aYNbdq0UVtZFchkSKVSsrKykMlk5OTkoG9goJRfYZnUVl1u5diU1NT5iy/YvWeP4tiZv/7K%0AV199RbWnekqijvw7dezIurVrn9XfR4+oUqVKmfqUtP57m3dr2VHbpnDrarc2H7Ln/GUlrY1r2bHD%0A2wMzIwNy86UkPc6gqnFhy8jEbzsytmd7AJLTs8iTFmBqaFA6o9fgHz09PQ4eOIBTo0aKOlO1DB/M%0ADwqiTevWtGnTpkx9p0+dwtm5WH11dWPv3tL1ujy7pKQkjh6JZPGSJaXSP3BgP9ZW1owZO65cP70K%0ALj8N4NSacC5Idld42hWJTFbwyn+awhvTRTJp0iQAtmzZwq1btxj/9I32ZXjw4AFPnjxBKpVy7949%0AatasWdEyVXLgwAGsrKxo3rz5c20TExKws7VVfLa1tSUzM5OsrCylpsfy7BITErCzs1P67fq1awDo%0A6emRkpJCb3d3Hj9+zOw5cxR2enp6hG7axJKlS7GxtqZp06bo6j6rRrY2NqW0JCQkYltcRzGbhMRE%0AZY02tly7fp3atWvxJDubk6dO8UmrVlyKjeXmrVskP0zGwMCAjz/+mCmTJ2FhYcGcuYH4+PoRtHBh%0ApfomLi4OKysrNm7YwJ8nTpCfl8eAAQOo/c47CtsbN25w5MgRdu7apdayqlWrFgMHDqRnjx6YmZlh%0AamrKhvXrFWkkJCZiWyLNUuVWjk1CYmIpTdeuXwcKr0GpVMq3337LH3/8QVmoK389PT02hYaydMkS%0ArG1s6NChQ5n6lLQ+ysC2mtmzvKqakZmTR1ZOnlI3iZ6ODpH/XMdv0z70dHUY8WXhw1lLSwtdHS0m%0Ar9vFob+v0uG9+rxja1F+npXon6I64967N48fPy7VSlFUh3ft3FmOvgRs7Z7VVxsV9bo8OxsbG+bN%0ADyozfVfXwi7AHdu3q3bSKxI60geARh3LX4VS3WhSS8Sr8sYEGGVx5MgRVq5cycaNG1myZAk5OTm0%0Aa9eOlStXoqenR3x8PF27dmX48OEAbN68mY4dO2JoaEhISAgTJxb26X/22Wd8+OGH3Llzh1atWpGR%0AkUF0dDR16tRh7ty5xMfH4+XlRUFBAVpaWkydOpVGjRrRunVrTpw4AcCYMWPo3bs39+/f59ixY+Tk%0A5BAXF4eHhwetW7dm69at6Onp4ezsXO7mMQAyFU2GOtraL2wnk8lKfa+t82wlTEtLSw4eOsSVK1cY%0A6uGBY926igdp7z59cO/dm2VLFrN7zx4+adWq3LTk8tJ5FdmUpUNHWwdTU1MWBM1nyZKlBAUtoGnT%0ApjRv3gw9PT3ea9KEBfPnKeyHD/uRjp99Tn5+fqX6RiqVcv/+fUxMTFi3bh1xcXH88P331Kpdm8aN%0AGwMQEhKCe+/eil0J1VVW9x884NChQ+zfv5+q1aqxMCiIad7eLF60CAB5GWmWTLc8m7LLTZsrV64Q%0AHhHB6lWryjz2RdKu7Pz79O5Nb3d3li5dynhPz+drVVE22tqlV7js8H59Orxfn80n/mH4snB2eQ9V%0A2P06sBvTen/O2D+28fvek4oApMw8K8k/RVhaWnLo4EGuXLmCx9Ch1HV05J2nrQwhISH0dndX1OGy%0A9amorzrar2QnKI0IMDSc9u3bc+LECSZOnEhCQgJr1qzhwoULPHjwgB07dpCXl0fbtm0ZPnw4MpmM%0AXbt2ERYWhq6uLl9++SWjRo3C0NCQ+/fvs27dOqytrWnRogXh4eFMmzaNjh07kp6ezpw5cxgwYACd%0AOnXiypUreHl5sWXLFpW6MjMzWbVqFXfu3GHYsGF88803fP3111hZWakMLhYuXEhkZCQyuZyszEzq%0A16+v+C0pKQlzc3OMjI2VjrG3s+NSsX7p4nb29vYkF+v3TEpKwtbWloyMDM6dPUuHjh0BcHJyokHD%0Ahly/cYOc3FzkMhmNnJzQ0tLi66+/Zt36DSQnJ5fKw7jYwDk7OztiYi6VaWNvZ8fD4sc/TMLW1gaZ%0ATIaxkTGr/ng2AKvnN99Qs2ZNLl68SHp6Bi4u7QBYtXo1UqmUvt99R1ZWVqX5xsbaGoCvevQAClsJ%0APvjwQy5dukTjxo0pKCjg8KFDdOnaFbengzDVVVbnzp6lnYsLFpaWALi7u/Ntr17PysTenphLZZfJ%0Ai9jY29srl9tTTTt37iQzM5OBAwc+Lc+HTPbyYuyYMbi4uKg1f3t7e2RyOU6NGinqb/ALjMGwszAn%0A5u7/PcsrLQNzY0OMDfQV38U9fERyehZNHR0A6NmqCdPDDpCenUNsXAL1q1thU8UMYwN9unzkxKF/%0ArpWfZyX5JyMjg7PnztHxacuNk5MTDRs04Mb167xTuzYFBQUcOnyYTWX4ZdmypRw7Wjj2JCsrk3pl%0A1Wsj5XptZ29HzKUy6nUJO0Fp3oSVPN/4MNLDw4Pdu3fTv39/RVN+gwYN0NXVxdjYGENDQwCOHz9O%0AVlYW48aNY9SoUchkMnY+bSKsWrUq1atXR09PD2NjY+rVq4eWlhZmZmbk5uZy8+ZNRdeGk5MTCQkJ%0ApXQUfwtq1KhwLX17e3vFoMXnMWrUKLZv345EImHDhg1ER0dz9+5doHC2QPGbdxGtWrVSaefi4sK2%0AbduQSqWkp6ezf98+2rdvj46ODj4+Pvz1119AYXPpndu3adKkCdevXcPbx4fs7GwAdu7cRbOPPiI6%0AJoa7d+MACI/YrHjwK+lQYePi4sK27dsLdWRksG//ftq7tEdLS4ufRo4kNvYyAAcOHkRXV5cGDerz%0A5Ek2s2bPVoy70NbRoWuXLkjCwyvVNzUcHHBycmLnjh0ApKSk8M/ffytaL65fv465uTmTJk1SDJZT%0AV1k5OTnx5/HjPHnyBIBDhw8rBa8l8wuPiCilqzybkpr27d9P+/btmTBhAjt37FCcv421Nb/OnPlS%0AaVdW/teuX8fH2/tZ/d21ixYtWpQqi1Jl0+gdou884G5SaqGOP//GpUk9JZvktEwmrt3Bo8xCf+85%0Ad5l69lZUNTHiwMV/Wb73JHK5nLx8KQf+ukqL+rXKz7OS/FNWnbl9545i5lFRHa5Ro0YpTSNG/ESY%0AREKYRML6DRuIKV5fI1TX6xexE7yZvNEtGAA+Pj5MmTKFxYsXKwYMammVbtqMiIhg+vTpisp/4cIF%0Apk+fjqura5n2xXF0dOT8+fN07NiRK1euYGVlBaAYZKenp8eNGzcU9mWlp6WlVWazZllYWFri5++P%0A5/jx5Ofn4+DgwPQZMwCIjY3Fz88PiURSrp2rmxv34uNxc3UlXyqlV69eNGvWDICgBQuYO3cuUqkU%0AfT09fv31V2xtbenWvTtx9+7R97vv0NHRwbFuXWbOmM6lS7GM9/QkX1qYx4yAAGJjL+Pn748kLBRL%0ACwv8fX1L2QC4ufYiPv4eru69kebn06vXtzRrVrhN8KyZM/ELCCA/Px9rKysWzJ+PlpYWbdq0pk+f%0APgz8/ntkMjn169XDx3vaa/HN/KAgfp05k/DwcORyOT/++CPvvvsuAHFxcVSvXl0jyqpHz548ePCA%0APr17o6+vj729PQH+/gpdlhYW+Pv5FZbJ0/xmTJ+upEmVTWG5uRJ/7x6ubm5Py+2ZphdBHfl379aN%0Ae3FxfNe3b2H9dXTE18fn+VrNTPDv24Xxq7aTX1CAg1VVZvT/kti4/8MvZD+SSYNoWq8mHp+3YvCi%0AUHS1tbGuYkqQx9cAjPu6PdPDDvDtr2vQAtq/V5++LuVrrUz/LAgKUtQZPX19RZ2BsutwWVhYWOLr%0A54+n53ikT/MOmP6sXvv7+REmkZRrJyifN6GL5I3bTbX4IM9169Zx7do1ZsyYwZYtWzhy5Aj9+vUj%0ANDSUoKDCwUWtW7dm+/btfPPNN0RGRioNWOzatSvTp09n5MiRirEUxcdV9OjRgz/++IPc3FymTZtG%0AXl4eUqmUqVOn0qRJE5YuXcrevXtxcHBAJpMxePBg7t+/r9CXm5tLly5diIyM5OjRo8yZMwdvb29a%0Atmyp8vzEbqqqEbupqkbspvocxG6qKhG7qZZPZe2mWu3z5we/qnh0wK8Clbw6b1yA8aYjAgzViABD%0ANSLAeA4iwFCJCDDKp7ICjKqdpr7ysY8PTa9AJa/OG99FIhAIBALB/xqqZgn9LyECDIFAIBAINIw3%0AYQyGCDAEAoFAINAw3oQA442fpioQCAQCgeD1I1owBAKBQCDQMDRpT5FXRQQYAoFAIBBoGG/CSp4i%0AwBAIBAKBQMN4E8ZgiABDIBAIBAINQwQYAoFAIBAIKpw3IcAQs0gEAoFAIBBUOGKpcIFAIBAIBBWO%0AaMEQCAQCgUBQ4YgAQyAQCAQCQYUjAgyBQCAQCAQVjggwBAKBQCAQVDgiwBAIBAKBQFDhiABDIBAI%0ABAJBhSMCDIFAIBAIBBWOCDDeIu7cucOxY8dISEhAE5Y/yczM5P79+2RnZ6tbikaxb98+pFKpumVo%0ALAkJCUqfb926pSYl8ODBA6W/pKQk8vPz1abnyZMnJCQkkJyczNKlS7l//77atAgEYqGtt4SNGzdy%0A8OBB0tLS6NmzJ3FxcXh7e6tFy7Zt2wgJCeHx48dYWFiQkZGBubk53333Hd27d1eLptOnT9OyZUu1%0A5F2SwMBAoqKiaN26Nb169cLR0VGtei5evIifnx8pKSnY2Ngwffp0Gjdu/Np1XLt2jcTERAIDA/H0%0A9ASgoKCA+fPns3379teuB6B79+4kJiZSp04d7ty5g5GREVKpFE9PT3r06PHa9QwZMoTevXtz4MAB%0A6tWrx5kzZ1i1atVr11GcI0eO0L59e8XnPXv20LVrV7VoiYmJoUmTJorPZ8+epUWLFmrR8jYg9iJ5%0AS9i9ezfBwcEMHDiQQYMG8e2336pFx6RJk2jatCl//PEH5ubmiu8zMjLYuXMnnp6ezJ0797XrWrx4%0AscYEGOPHj2fs2LFERUWxYMECHj58iJubG927d0dPT++165k+fTrz5s2jXr16XLt2DW9vb0JDQ1+7%0AjvT0dPbs2UNKSgq7d+8GQEtLi+++++61aynCwcGBdevWYWFhQVpaGlOnoJ8TxQAAIABJREFUTiUg%0AIAAPDw+1BBg5OTl07NiR9evXM2fOHE6ePPnaNRRx5MgRLl68yO7du/nrr7+AwoAwMjLytQcY58+f%0A58aNG6xdu5bvv/9eoSUkJIRdu3a9Vi1vEyLAeEuQy+VoaWmhpaUFgL6+vlp0+Pn5YWBgUOp7MzMz%0AvvvuO7UFPlpaWvz000/UqVMHbe3CnsOxY8eqRYtcLufPP/9k27Zt3L9/n6+++opHjx4xbNgwtbyN%0AmpmZUa9ePQAaNGiAoaHha9cA0KxZM5o1a0ZsbCzOzs5q0VCSlJQULCwsAKhSpQrJyclUrVpVUYde%0AN/n5+axbtw5nZ2du3Lih1u7HRo0a8fjxYwwMDKhTpw5QeJ1169bttWsxNzcnOTmZvLw8Hj58qNBS%0A1BImqBxEgPGW8OWXX9K3b18ePHiAh4cHnTp1UpuWjRs3YmBgQI8ePRSBTmhoKL179y4z+HgdqCuw%0AKYvPP/+cZs2a0b9/fz766CPF9zdu3FCLHktLS6ZMmULLli2JjY1FJpMRFhYGgLu7+2vX8/jxYzw8%0APMjNzVV8t379+teuA8DZ2ZmxY8fywQcf8Pfff+Pk5MSePXuwtLRUi54JEyZw+PBhhg8fzo4dO5gy%0AZYpadADY29vz9ddf06NHD7UFXEU0aNCABg0a4Orqiq2trVq1vE2IMRhvETdv3uTatWvUrVuXhg0b%0AqkXDqFGjqF27NlKplLNnz7Jq1SqqVKnCgAED1PaQAJBKpWzdupUHDx7QsmVL6tevr3gzfd1kZmZi%0AamqqlrzLYsmSJSp/+/nnn1+jkkK6deuGl5cXdnZ2iu/q1q372nUUcfjwYW7evEmDBg1wcXHh1q1b%0A2NvbY2RkpBY9KSkpSsFX9erV1aKjiN9//52VK1cqtXz9+eefatGybds2fv/9d/Ly8hStuocPH1aL%0AlrcB0YLxljB58mTF/1FRUejp6WFnZ0ffvn2pUqXKa9ORmprKwoULAThw4ADDhw9n7dq1ap/V4uPj%0Ag42NDSdPnqRJkyZMnDiRlStXvlYNbdq0Ufmbum7IAN98802p79T50LK3t+eTTz5RW/7Fefz4MdnZ%0A2djY2PDo0SN+//13fvzxR7Xp8fX1JSoqChsbG8UDVB3jZYqze/dujh8/rraAqzgrV65k+fLl2Nvb%0Aq1vKW4EIMN4ScnNzqVmzJs2aNeOff/4hJiYGCwsLJk6cyPLly1+bjvz8fFJTU7GwsODzzz/nwYMH%0AjB8/Xq1T+wDi4uKYMWMG58+fp0OHDqxYseK1a1BnEFEeY8aMQUtLC5lMRnx8PLVr12bTpk1q02Np%0AaYm3tzeNGzdWjClSR1cNFLbg1K1bl2vXrmFgYKD2h2h0dDSHDh1Se5dEcRwcHNQ2bqckNWvWpHbt%0A2uqW8dYgAoy3hNTUVObPnw9A27Zt+eGHHxg9ejR9+/Z9rTpGjRpF37592bBhA1ZWVgwaNIjs7Gwi%0AIyNfq46SFBQUkJqaipaWFpmZmWq9QZ88eRKpVIpcLicgIIBRo0apbfouoBhvAYUzOaZNm6Y2LVD4%0AwAJITk5Wqw4oHJDr7+/P5MmTmTFjhlpntADUrl2b3NxctQc6xcnPz6d79+40aNAAKBxcOW/ePLVo%0AMTQ0ZMiQITg5OSmCU3UN5n4bEAHGW0JmZiY3b97E0dGRmzdv8uTJEx49esSTJ09eq45WrVqxd+9e%0Ape+GDx+Om5vba9VRktGjR9OnTx8ePnyIu7s7Xl5eatMSFBTEvHnz8PPzY9OmTYwePVqtAUZxzMzM%0AuHfvnlo1lNVloy50dHTIzc0lOzsbLS0tCgoK1Krn//7v/2jfvr3iLV0Tukg8PDzUmn9x2rVrp24J%0AbxUiwHhL8Pb2xtPTk6SkJAwNDfn666/Zs2cPw4YNU4ue0NBQQkNDycvLU3y3Z88etWgBaNGiBfv3%0A7yc1NZVq1aop3m7UgaGhIZaWlujq6mJtba1WLVDY/aClpYVcLic1NVXt4x80qcumb9++rF27ltat%0AW9OuXTulWT/qQF0tA+Xx4MEDdUtQUNT6JXg9iADjLeG9997D19eXjRs3cuLECVJSUvjpp5/Upmf9%0A+vWsWLHitQ4wLYv+/furfICra1aLiYkJQ4YMwd3dneDgYLXNZimiqGsNwMDAACsrKzWq0awumy++%0A+ELxf5cuXdQ++0dHR4eZM2dy8+ZN3nnnHaXB3eri5s2bQGF30pUrV6hatSo9e/ZUi5aiQFQul3Pj%0Axg1q1KhB8+bN1aLlbUAEGG84eXl5ilU89fX1yczM5PDhw2ofdNWwYUPs7e3R0dFRqw4/Pz8Ali5d%0ASseOHfnoo4+Ijo7myJEjatO0aNEi4uLiFCtnurq6qk0LlP3Q0pQ3QXV12RS16pSFOrskpk6dSp8+%0AfWjevDlnz55lypQprFu3Tm16AMaNG6f4Xy6Xq3WWTfFgOS8vj9GjR6tNy9uACDDecDp06EC3bt0I%0ADAzknXfeYciQIWoPLgBatmxJp06dqFmzpmI6nTpaDIrWT0hOTlYsX/zZZ5+xYcOG166liLJmsKhj%0AvYkiNO2hpQldNsUfVJpEbm4uHTt2BKBTp06sWbNGzYpQ6gZ9+PAh8fHxalTzjIKCArWPJ3rTEQHG%0AG87AgQPZuXMn9+/fp1evXmpfb6KIsLAwFixYgJmZmbqlKAgPD+e9997jr7/+UsueH0UUdUHI5XIu%0AX76MTCZTmxYo/dBau3atWvVoQpfNqlWrFJsFXr58WS2bv5VFQUEBV69epWHDhly9elXt43cAOnfu%0ArPjf0NCQwYMHq01L8bVmpFIpAwcOVJuWtwERYLzheHh44OHhwdmzZwkPD+fSpUvMnTuXHj16KKaN%0AqQNbW1uaNGmiMfP1AwMDWb58Ofv27aNevXoEBgaqTUvv3r2VPg8ZMkRNSgop+dBSN5rQZVN82fZZ%0As2apdRXa4kydOhUvLy+SkpKwtbUlICBA3ZIUU9BTUlKoVq2aWq95TV1r5k1FBBhvCS1atKBFixak%0Ap6ezfft2JkyYwLZt29SmJy8vjx49elC/fn3FW5Y6R8BbW1szYsQIxRLL2dnZVKtWTS1abt++rfj/%0A4cOHah+FP23aNLy8vHj48KFiu3Z1ogldNsVbAjWlVRCgcePGbN68Wd0ylDhz5gxTpkzB1NSU9PR0%0AAgICaN26tVq0XL16FS8vLxITE7GysmLmzJka0/r0JiICjLcMc3Nz+vfvT//+/dWqQ50DvcpCk5ZY%0ALmp6h8IugIkTJ6pFRxEnT57UqIeWJnTZFO960IRuiF9++YVFixaVudy8ut/aFyxYQHBwMLa2tiQm%0AJvLzzz+rLcCYPn06M2bMoFGjRly5cgU/Pz+1rxPyJiMCDIFaSEpKUmzbnJSUhJeXFy1atFCbHk1a%0AYnnDhg08evSIe/fu4eDgoPZpqseOHWPQoEFqn/FThCZ02Vy8eFHxMH/8+LHSg10dD/RFixYBheOI%0Aiu+zUTRFVJ3o6OgodjC1tbVV247JRTRq1AgAJycndHXFI7AyEd4VqIXt27djYmJCXl4e8+fP55df%0AflGrHk1aYnnv3r0sWLAAR0dHrl+/zs8//0yPHj3UpufRo0e0bdsWBwcHtLS01L46pCaMM7h06dJr%0Az7M8rl27RmJiIoGBgUyYMAG5XI5MJmPevHls375drdpMTU3ZsGEDzZs359y5c2pd+0ZbW5sjR47Q%0ArFkzzp07h76+vtq0vA2I7doFaiEnJ4dhw4aRm5vL0qVL1f6W3rt3b+7cuaMRSyy7u7uzevVqTExM%0AyMzMZODAgWrtorh//36p72rUqKEGJQJVnD9/ns2bN3P8+HHatm37/+3de1TN+f4/8OeuVOSSS2Vh%0A2yq5Dblt9/th6HRGLkP2lAZj3HJJGSpkEsLM4GQlnXMGk0oXaTqmMihj5ZZ0cHJoZqOQUKhkR7W1%0AP78/mr2/tsucdX5rfF4f0+ux1qxl7/7YzzUun9d+v9+v1xtA/Z/h3r17k10Ep/f06VNERESgoKAA%0Ajo6OWLBgAVmRUVxcjK1btxqyrFq1iv8sv0NcYDBR+fn5Gfasq6qqkJOTgzFjxgCgPeQppYeoSqUy%0AKm48PDxw4MABkixnz57F0KFD8dVXX6G8vBwymQwrVqxA69atSfIA9Xe1vFpwUZ8zkIqrV6+idevW%0AaNu2LfLy8uDs7EwdCXV1dbh+/TpqamoMf/cpc2k0GlRXVxuyUP5Z/qPjLRImqldbMD/77DOiJMbM%0AzMzw9ddfo6ysDC4uLujatStZgSGXy7FlyxYolUrk5uaiY8eOJDkiIiJw/fp1DB06FLm5uVi6dCly%0Ac3MRERFBOp775MmTOHHiBC9vv0FCQgIUCgXmzp2Lw4cP4/Dhw1i7di1ppvnz56O2thYtWrQwHKAO%0ADw8nybJq1SpcvHgRzZo1M2T5/vvvSbI0BFxgMFFRHuT8LUFBQZgzZw4iIiKgVCoREBCAxMREkiyb%0AN29GQkICzp49C0dHR6NRy2I6d+6coUPDwsICI0aMwNChQ8lHl/fo0QM1NTWSKDDy8/ORkJBgaG8G%0A6n//qFy7dg0hISEA6s+qeHp6kmXRq6mpQUxMDHUMAPUt4BkZGdQxGgwuMBhD/ZmQIUOGYPfu3XBw%0AcCA56X7lyhX06tUL2dnZUCgUhvMg58+ff2P7oRj0nSP6iYempqbk01ednJwwfPhwtGnTxvAtNDMz%0AkyRLQEAAZs6cibZt25J8/puUl5ejZcuWqKysJL8+HgCUSiVOnToFR0dHw3vt2rUjyeLs7IyCggLD%0AFQHs3eICg4nq/PnzUCqVkml51LOwsMCpU6eg0+lw+fJlkm/H586dQ69evZCWlvbazygKDK1Wi9ra%0AWpibm2PcuHEA6gekUT+00tPTkZmZiebNm5PmAOrHulOv6Lxs8eLF+Pjjj9GiRQs8ffrUaKYKlceP%0AHyM0NNTw+0V5gLpp06aYNm0amjRpYniPz++8O1xgMFHl5+cjJiYGjRs3xrBhwzBq1ChYW1tTx8KG%0ADRsQEBCAq1evYseOHdi0aZPoGebPnw+g/qbZKVOmkF9lP3HiRKxevRpBQUFo0aIFKisrERoaaphf%0AQqVdu3Zo3LixJLZI2rdvj7///e/o3r274dAg1WoTAIwZMwYjR45EeXk5WrduLYkhYAUFBThy5Ah1%0ADAD1X3BycnJ4/oVI+P8yE9Xs2bMxe/ZsaDQanDp1Clu3bkVlZSV69+5teMCK6caNGwgJCcH+/fvx%0A4MEDdOnSBbdu3cK1a9fIriTX6XSYM2cO7O3t4e7ujkGDBpHk8PT0hEwmw8yZM/HkyRNYWVnB09Pz%0AtYO6Ynvw4AE+/PBDyOVyALTfiLVaLQoLC43Gu1MWGJmZmThw4AC0Wi0EQUBFRQV++OEHsjxAfcF8%0A+fJlo5HcVMVhp06d8PjxY8PgL/ZucZsqE9Xz589fG2YlCAIuX76Mvn37vvHn79LChQuxePFi9OrV%0AC15eXoiOjsbt27exdu1a0ivbgfrponv27MHPP/+Mo0ePkmaRkldbih88eID+/fsTpTFWWloKW1tb%0Ass+fOHEiQkJCEB8fj0GDBuHs2bOkF/fpM1VVVRleU56ZGT9+PIqLi43uGeItkneHVzCYqEJCQtCz%0AZ0+4uroa/pLLZDIoFAp89913yM/Px9atW0XL8/z5c/Tq1QsADIcXFQoFXrx4IVqGV1VXV+Po0aNI%0ASUmBIAhYunQpWRYp0rcPZ2dnIzY2FhcvXsSZM2dIsoSFhSEuLg5arRbV1dXo1KnTG8/QiMXW1hZ9%0A+/ZFfHw8pk6dKokWzJdXULRaLWmxfOzYMaPXly5dIkrSMHCBwUS1efNmpKenY/HixXjw4AGsra1R%0AVVUFGxsbeHh4YPbs2aLmebm9MCIiwvBryj1aNzc3TJgwAcHBwYZOElbv2bNn+P777xEXF4eHDx8i%0AKCiIdEDbiRMnkJWVhdDQUMyZMwfr168nywIAjRo1woULF/DixQucOnUK5eXlpHn0SktLER8fj0OH%0ADqFbt26k53hqa2vxww8/IDY2FrW1tUhNTSXL8kfHBQYTnaurK1xdXVFTU4MnT57A2tqabE/W1tb2%0AtYmHeXl5sLGxIckD1HdJFBUV4datW7CwsICdnR35Yb1bt27h9u3b6Nq1K1meDRs2IDs7G+PGjUN4%0AeDg2btxIfuDUxsYG5ubmqKqqgkKhgFarJc2zfv16FBQUYNGiRQgLC8OiRYtI8+Tk5CAmJgb5+fkw%0AMTFBfHy80WVsYrp79y5iY2Nx5MgRCIKAHTt2oF+/fiRZGgouMBgZCwsL0v1qAFi5ciW8vb0xePBg%0AKBQKFBUV4dy5c4iMjCTLFB8fj+PHj+PJkyeYPHky7ty5Q9puGBMTI4k8//rXv/DBBx+gd+/e6Nix%0AI3nRBQBt27ZFUlISGjdujG3btqGyspI0j52dHaysrCCTyTB27FjSA6dTp06Fg4MDVCoVBg8ejPnz%0A55MVFwsXLoRGo8GkSZOQmpqK5cuXc3EhAvq7qRkjJJfLcfDgQfTt2xfPnj1Dz549ER8fTzYICADS%0A0tKwb98+NGvWDLNnz8a///1vsixSypOSkgKVSoXjx4/DxcUFt27dIr+OPCQkBEOHDsWqVatga2tL%0Aul0DAL6+vsjMzMTXX3+NixcvYvXq1WRZnJ2doVarkZWVhcLCQvKC0NTUFNXV1dDpdORZGgpewWAk%0A9FMrpcDS0hKurq7UMQz00yn1/whSz3uQUp5+/fqhX79+0Gg0OHz4MFauXAkASE5OJsnz7NkzREdH%0A4+bNm+jUqRMmTpxIkkOvtLQUkyZNQlJSEqKjo0U/0/Sy4OBgVFdX48iRIwgKCsL169dx4MABuLq6%0Aij77JjIyEvfv38ehQ4cwffp0PHv2DFlZWRg+fDhMTPh79rvCbaqMhK+vL4qLi+Hm5gY3NzdJTGWU%0AipiYGKSnp+PevXtwcnLC4MGDMXfuXM7zFteuXTOasSCmZcuWQalUYsCAAcjJySHfXnN3d8fnn3+O%0Ac+fOYenSpViwYAEOHjxIludlN2/eRFJSEtLS0pCVlUWWQxAEZGVl4dChQ8jLy8PJkyfJsvzRcYHB%0AyDx58gSpqanIyMhAq1atSIdKSUlBQQEEQYBarYa9vT26detGHQk3b96EWq2Gg4MDunbtSh1HMvSz%0AU/Q8PDxw4MABsjzHjh1DWloaAgMDkZCQAGdnZ4wZM4Ysj55Go4FMJsPx48cxYsQIyVyR/vjxY8lk%0A+SPiAoORuXnzJpKTk3HmzBkolUrodDpUVlaSDwai9sknnyAuLo46hkFeXh7S0tKMWnqDg4PpAkmI%0Au7s7du3aBRsbGzx69AhLliwhmyoqVb6+vhg9ejQuXboEnU6Hx48fY9euXSRZIiMj8e2338LS0tLw%0AHg/aenf4DAYjMX36dFhaWsLd3R0+Pj6GfX0pLb1TadKkCUJDQ2Fvb2/YH54xYwZZHn9/f8ybN4+3%0Asd7Ax8cHKpUKzZo1g0ajwYYNG0jzSPEBKqVzIenp6Th16pSo04IbMi4wGImgoCCj2RM5OTkYOHAg%0A9uzZQ5hKGvr27QugfvlWChQKBaZOnUod440tl1VVVaiurkZ+fj5BImDYsGHIzMxEWVkZWrVqhdu3%0Ab5Pk0JPiA1Sr1eLYsWPo3LkzysrKjMaGi61Dhw5GxRd7t7jAYKLKzc3FjRs38N1332HOnDkAgLq6%0AOhw4cIAn6v1q8eLFyMjIQGFhIZycnMj30CdMmABfX184Ojoa3luyZInoOV79Jh4XF4e9e/ciICBA%0A9CyvatWqFQBgxYoVSEpKIsshxQfovHnzkJqaisDAQERHR2Px4sVkWbRaLSZOnIguXboYuqKoW4v/%0AyLjAYKJq3rw5Hj16hNraWjx8+BBA/V0k+nZDBqxduxbPnj1Dnz59kJKSguzsbAQGBpLliY2Nxfjx%0A4yWzRVJSUoI1a9bAysoKCQkJhoe7FFAfaXv5AQrU/92ifoBWVFQgLCwMQP2W0v79+8myzJs3z+g1%0Az8N4t7jAYKLq0qULunTpAnd3d/IpnlKlVqsNrYWzZs2Cu7s7aR5ra2vMnz+fNIPeP//5T4SHh8PH%0Ax4d8TPibUD+wXn2AUkpNTcWJEydw/vx5ZGdnA6hfrbx+/To+/fRTkkxZWVnw8/ODiYkJKisrsXbt%0AWgwYMIAkS0PABQYT1bJly7Bz58437ulTH0aTio4dO6KoqAhyuRyPHz8mG6+s17JlS6xbtw49evQw%0APEApDp0uXboUFy9ehJ+fH6ytrY3+vIg9EtvPz++1YkIQBBQVFYma41VdunTB6dOn8eLFCwiCgNLS%0AUgwcOJAky4gRI2BjY4OKigqoVCoIggATExPI5XKSPED9kLjZs2fj008/xc6dOw3btOzd4DZVxiRm%0A3LhxKCkpQbt27fDgwQOYm5vDwsICAE0RFh4e/tp7FGcwfmubaPPmzSImqT+U/DZUD3QAmDlzJhwc%0AHKBWq2FhYYHGjRuTDv4CgNu3b+PKlSv46KOP8M0330ClUqFDhw4kWQRBgL+/P9LS0rBmzRp4eHiQ%0A5GgouMBgJM6ePWv4lrVhwwb4+PiQj1lmb3fy5Elcv34d9vb2GDduHEmG0tLSN26rXbhwgZe5f+Xp%0A6YnY2FgEBgZi06ZN8PDwIJ/LoVKpEBAQgD59+uDChQsIDw9HVFQUSRZPT0988MEHmDt3Lr788kvY%0A2NiQtxb/kfEQdkZix44d6NSpE/bv34+4uDjyfwTZ223btg3Jyclo1KgRUlJSsHXrVpIcc+fOxblz%0A5wyvBUHAzp07sXbtWpI8UmRqaoqamho8f/4cMpkMdXV11JEAAH369AEADBgwADqdTvTPX758OYD6%0AMyqrV6+GnZ0dIiMj4eTkJHqWhoTPYDASlpaWaN26NczMzGBjY0N+OI693YULFwwFIOWh02+//RZ+%0Afn64dOkSpk6dii+++ALt2rXDoUOHSPJIkaenJ6KiojBs2DCMGjUK/fv3p46E5s2bIyEhAX369EFe%0AXh6srKxEz1BWVgYAGD16tNH7VIdNGwouMBiJpk2b4vPPP8eMGTMQGxsrqVZDZuzFixfQ6XQwMTEx%0A3KxKwc7ODlFRUVi0aBEiIiKwcuVKzJo1iySLVNXU1Bg6fv785z+jadOmxImALVu2YPfu3Th+/Dg6%0Ad+6M0NBQ0TMUFRVh+/btb/yZn5+fyGkaDi4wGImwsDDcuXMHnTt3hlqtxvTp06kjSUZKSgr+9re/%0Aoba21vBAz8zMJMvj6uqKTz75BL1790ZeXh7Z1fa1tbUIDQ1FRUUFvvzyS+zZswcODg4YMWIESR4p%0ASkxMhJubGwBIorgAgOrqaqNujerqatEzWFpawt7eXvTPbej4kCcjcf/+faSmphpdoEXRmSBFf/nL%0AXxAREWHUnqq/q4WKWq1GQUEB7O3tyW5TdXNzw8iRI7F8+XKYmZnh7t278PPzg1KpxKpVq0gySY27%0Auztqa2uN7rGhHrQ1Y8YMyGQy6HQ63L17FwqFQvTL/F699ZaJg1cwGAkfHx8MGTKEfMaDFMnlcigU%0ACuoYBomJiSgsLIS/vz8+++wzuLm5YfLkyaLnCAgIwNChQw2vO3TogJiYGHz11VeiZ5GqL774gjrC%0AaxISEgy/rqysRFBQkOgZevbsKfpnMl7BYETmzJmDffv2UceQpOXLl0Oj0aB79+6G8w6U+8RTpkzB%0AwYMHYWZmBq1Wi5kzZxo9NMSyceNG7hh5i+XLl+Ovf/0rdYz/ShAEfPzxx0hOTqaOwkTAKxiMhJOT%0AE9LS0oweorxHWm/UqFHUEYyYmJjAzKz+n4pGjRqRHfJUq9Ukn/s+0HdJSJF+i0QQBJSVlWHIkCHU%0AkZhIuMBgJPLz842u2JbJZKSXIEnJxIkTceXKFaNxz5TGjh0LDw8PODs74+rVq/jTn/5EkqOkpOSt%0AKycUo8ulRMpdEi/nsrCwQJs2bQjTMDFxgcFIREdH4+nTpyguLoZcLifpjZeqJUuWQKvVorS0FHV1%0AdbC1tSW92Mvb2xtjxoxBYWEhJk+ejG7dupHk0Gq1hht4mTEpdkm8acS8Hh/obhi4wGAkjh49it27%0Ad6Ourg4uLi6QyWTw9vamjiUJ5eXlSEhIwJo1axAUFER+IdP9+/dx+vRp1NTUoKCgABkZGSQPiPbt%0A2/OD6S3atGmDKVOmUMcwol+pyMjIQIcOHdCvXz9cuXIF9+/fJ07GxMKjwhmJffv2ITExEdbW1vD2%0A9kZGRgZ1JMmwtLQEADx//hyWlpbkU059fHyg0WjQpk0bw38U7OzsSD73fSDFLgmVSgWVSgWdTofg%0A4GC4ublhzZo1qKqqoo7GRMIrGIyEqakpzM3NIZPJIJPJ0LhxY+pIkjF+/HiEh4ejW7ducHd3R5Mm%0ATUjzWFlZwdfXlzQDAHzzzTfUESTL39+fOsJbVVRU4M6dO+jYsSMKCgrw9OlT6khMJNymykhs374d%0Ad+/exdWrVzFo0CA0adIEAQEB1LEk55dffoFCoTCsalAIDQ1F7969ueOH/X/Jzc3F+vXrUVZWBjs7%0AOwQHB8PZ2Zk6FhMBFxiMTFZWFtRqNRwcHMg6E6Tol19+werVq1FSUoI2bdogNDQUPXr0IMvj5eVl%0A9Jo7ftj/qry8HEVFRejQoQPfO9SAcIHBRPfzzz/j6NGjKC8vR9u2beHi4oJOnTpRx5IMLy8vrFmz%0ABt26dUN+fj7Wr1/P19kDqKurQ11dHfz8/LBjxw4IggBBEDBv3jwueH6Vl5eHtLQ0oxH8wcHBdIEA%0ApKenIywszHDv0JIlSzBp0iTSTEwcfAaDierIkSP4xz/+AZVKhZ49e+LevXtYtmwZli1bhnHjxlHH%0Akwx9K2j37t0NQ67Eph+Q9CYUBc+hQ4cQGRmJR48ewcXFBYIgwMTEBEqlUvQsUuXv74958+ahefPm%0A1FEMoqKikJycDCsrK2g0GsyaNYsLjAaCCwwmqv379yMmJsbo4OKUKVOwaNEiLjB+ZWJigp9++glK%0ApRIXLlwgu+jsbYObqLi7u8Pd3R1JSUmYNm0adRxJUigUmDp1KnVyePLCAAAFbklEQVQMIzKZzDDn%0ApmnTprCwsCBOxMTCBQYTlZmZ2WtdEU2bNoWpqSlRIukJDQ3F1q1bsW3bNjg6OmLjxo0kOdq3b0/y%0Auf9Nz549cenSJZiYmGD79u1YuHAhj5/+1YQJE+Dr6wtHR0fDe9SzQ+RyObZs2QKlUonc3Fx07NiR%0ANA8TD8/BYKJ625K7TqcTOYl0tW/fHjt37kRqairCwsJw+fJl6kiSEhwcDHNzc+zevRu+vr6/OTGy%0AoYmNjUX37t3JZ5a8bNOmTZDL5Th79izkcjk2bNhAHYmJhFcwmKhu3LiBFStWGL0nCAJu3rxJlEj6%0A9u7dC1dXV+oYkmFubg4nJydotVr06dMHJib8PUnP2toa8+fPp45hZOHChdi7dy91DEaACwwmqrdd%0AKa1SqURO8v7gRi9jMpkMq1atwsiRI5Geno5GjRpRR5KMli1bYt26dejRo4dhtZD6IrjmzZsjIyMD%0A9vb2hmKQ56g0DFxgMFENHDiQOsJ7h3pUuNTs2LEDV65cwciRI3H+/HnJHUalpFAoAACPHj0iTlJP%0Ao9GgqKgIUVFRhvd4jkrDwXMwGJOI4cOHv/H9iooK/Oc//xE5jXRVVFTg9OnTRtfZL1iwgDqWJNy7%0Ad++199q1a0eQBIiJicHevXthamqKoKAgjBw5kiQHo8MrGIxJxOnTp6kjvBeWLFkCBwcHqNVqWFhY%0A8D02L/H19YVMJoNOp8Pdu3ehUCgQFxdHkiU1NRU//vgjNBqNYUuLNSx8Ooox9l4RBAEhISGwt7fH%0Avn37UFFRQR1JMhISEhAfH4/ExET8+OOPsLW1Jctibm4Oc3NztGrVClqtliwHo8MFBmPsvWJqaoqa%0Amho8f/4cMpkMdXV11JEkqVmzZigqKqKOAYAPKjdUvEXCGHuveHp6IioqCsOGDcOoUaPQv39/6kiS%0AoR/vLggCysrKSAeQ6VvSBUF4rT1927ZtZLmYePiQJ2MS4eXl9daOET51/2YajQaPHj3iy/J+VVxc%0AbPi1hYUF6aCtnJyct/6Mu8kaBi4wGJOIgoICAMCuXbswduxY9O/fH3l5efjpp58QGhpKnE66pk2b%0AhqSkJOoYpFJSUt76s8mTJ4uYhLH/w1skjEmEg4MDgPoZBvrJnR9++CGio6MpY0kef0fCa5NwBUFA%0AcnIyLC0tucBgZLjAYEyCDh48CGdnZ1y6dIknVf4XPIgMRucb7ty5A39/f4wePRqrV68mTMUaOt4i%0AYUxiHj58iMjISNy6dQudO3fGwoUL0bJlS+pY5Pz8/F4rJgRBwJkzZ3D+/HmiVNISGxuLqKgoBAYG%0AYsyYMdRxWAPHBQZjElFYWGj0WhAEwwOV727gQ4O/paSkBIGBgWjRogWCg4PRokUL6kiMcYHBmFR4%0AeXkZvda3G/LdDey/USqVMDc3x+DBg19b5eGWUEaFCwzGJOjp06coLi6GXC6HlZUVdRwmcby6w6SI%0ACwzGJObo0aPYvXs36urq4OLiAplMBm9vb+pYjDH2P+FR4YxJzL59+5CYmAhra2t4e3sjIyODOhJj%0AjP3PuMBgTGJMTU1hbm4OmUwGmUzGt4Uyxt5LXGAwJjH9+/eHn58fSkpKsG7dOvTq1Ys6EmOM/c/4%0ADAZjEpSVlQW1Wg1HR0eeZ8AYey9xgcGYRPB9EoyxPxIeFc6YRLx8n0RaWho++ugjo2FbjDH2PuEV%0ADMYkyMvLiy85Y4y91/iQJ2MSxKsWjLH3HRcYjDHGGPvd8RYJYxKhvy1UEARkZ2djyJAhhp/xfRKM%0AsfcNFxiMSQTfJ8EY+yPhAoMxxhhjvzs+g8EYY4yx3x0XGIwxxhj73XGBwRhjjLHfHRcYjDHGGPvd%0AcYHBGGOMsd/d/wNwQw9onEuxlgAAAABJRU5ErkJggg==%0A)

Let's define the attributes for fitting and the class to be predicted.

In [296]:

    X = JoinTraining[JoinTraining.columns.difference(['County','Client ID','Loan Flag'])]
    Y = JoinTraining['Loan Flag']

Define the train and the test with a ratio of 0.8 and 0.2 of the initial
data

In [297]:

    from sklearn.model_selection import train_test_split
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y, random_state = 42, test_size = 0.2)

Import the models that we will test. I usually give always a try to
Logistic Regression, Decision Tree (and Random Forest). However, since
in this case the records are not so many and the class to predict is a
binary label, I have quite high expectations even for Linear SVM.

In [298]:

    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC, LinearSVC
    from sklearn.ensemble import RandomForestClassifier

    models = []
    models.append(("LRegression",LogisticRegression()))
    models.append(("NaiveBayesian",GaussianNB()))
    models.append(("KNN",KNeighborsClassifier()))
    models.append(("DTree",DecisionTreeClassifier()))
    models.append(("RForest",RandomForestClassifier()))

We will try all these models with cross validation (10 splits)

In [299]:

    results = []
    names = []
    for name,model in models:
        kfold = KFold(n_splits=10, random_state=42)
        cv_result = cross_val_score(model,X_train,Y_train, cv = kfold,scoring = "accuracy")
        names.append(name)
        results.append(cv_result)
    for i in range(len(names)):
        print(names[i],results[i].mean())
        
    ax = sns.boxplot(data=results)
    ax.set_xticklabels(names)
    plt.savefig('AccInitial', bbox_inches='tight')
    plt.show()

    ('LRegression', 0.99160746045338632)
    ('NaiveBayesian', 0.9887257254525551)
    ('KNN', 0.98133631952220968)
    ('DTree', 0.99386184484992202)
    ('RForest', 0.99173324424954745)

![](data:image/png;base64,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%0Avviijh49qqNHj2r16tX63e9+p7/+9a9yu93Kzs7WpEmT9Oqrr6qqqkqhoaEaOXKkFi5cqP/+7//W%0Ayy+/LJNGb3NpAAAJnElEQVTJpB/84Af6zW9+o5UrV2rQoEGaPn26iouLtX37dknST37yE913331a%0AsGCBIiIitG/fPh08eFDFxcVKSUnxy/YaN26c3G63Xn31Vd19993en//6179WfX29jh49qhEjRmjp%0A0qV64YUXNGjQIH366acaMWKE0tPTdejQIc2ZM0eVlZX69a9/fdo2+7auri45HA5dcsklPa4jKytL%0AhYWFuuqqq/TBBx9o8+bNmjdvnh577DHZ7XZJ0sKFC5WcnKy8vDx99tlnOn78uO69917ddddduvXW%0AW/XOO+/os88+U3Fxsbq6umS321VQUCCbzaYf/ehHstls2rt3ry655BK98MILhu8lVVVVaf369Xrl%0AlVcUExMjSXrkkUf0wgsvaNy4cbrssstOmf+uu+7Sxx9/rM2bN+uWW27xR8mGYjabNW3aNL377ru6%0A+uqrT5ve2tqqX/ziF4qNjdWECRM0YcKE0/Y70dHRPv8/+pOOjg4dPHjQ+/d6JmvWrNF//dd/yWQy%0AacyYMZo/f75eeOEFffzxx2pvb9eSJUv04Ycf6u2331ZISIgmT56se++994z75FWrVqmpqUnl5eWa%0ANm3aRWxpz/p1aOfm5spkMmn//v0aPXq0li5dKklatGiRioqKNGzYML3++uv63e9+p5EjR+ro0aP6%0Awx/+oLa2Nv3oRz/yLmfcuHHKzs7WBx98oNbWVr322ms6ceKEMjMzdeONN6qyslJPPPGERo0apbKy%0AMrlcLr399tuaPXu2Jk6cqKqqKjkcDu/yNm/erNbWVlVUVMjlcmnGjBkaN26cJOmf/umf9NRTT6mi%0AokLl5eV66qmnLu5G+5aCggJlZGRo/PjxkiSHwyGr1apXXnlFbrdbP/7xj3XgwAHv/BkZGXrqqaeU%0Anp6ut956S1OnTu1xm0nSsmXL9PLLL+vgwYOKjIzUiBEjelxHRkaG3nzzTT366KN64403NGfOHK1a%0AtUrjxo3TjBkz9OmnnyovL08vv/yytm3bpoqKCklSTU3NKW1qaWlRbm6ukpOTVV1drcrKStlsNn3+%0A+ef6/e9/r8svv1xZWVnauXOnRo8efZG2dN/761//qgMHDujLL79UV1eX9+eXXnqpfvWrX+mxxx5T%0AaWnpKZ8JCwtTcXGx/v3f/93Qbb+YLrnkEjU0NPQ4/dChQ3rjjTcUERGhzMzM0/Y7NpvtjP8fJzsY%0A/cFHH32ke+65R0eOHFFoaKgyMzOVmpqqL7/8Uvfcc493vtzcXIWHh+udd97Rhg0bZDKZNHfuXG3e%0AvFmSlJSUpIULF6qlpUV/+tOfVFZWJknKycnRTTfddMZ98s9//nNt2LAhYAJb6uehfXJ4fMOGDacc%0An929e7eefPJJSVJnZ6cSExNlNpu9O6r4+HglJSV5l3PllVdKkv7+97+roaHB+4fkcrm0b98+LV26%0AVGvWrNEzzzyj0aNHy+PxKC8vT6tXr9b69euVlJSk2267zbu83bt3a8yYMQoJCVF4eLiuu+467d69%0AW5K839gvu+wy/e1vf7vAW+js4uLilJ+fr9zcXNlsNg0YMEBffPGFHnnkEUVFRam9vV2dnZ3e+YcN%0AG6auri7t27dPf/rTn7R27VqVl5efcZtJpw6PL1++XMXFxSooKFBbW9tp65g0aZKmTp2q2bNn68CB%0AA0pJSdHzzz+vjz76SO+8844k6csvv5TFYlF+fr4WLVokh8OhKVOmnNKmH/zgB3rppZc0YMAAOZ1O%0A7/HbuLg479/H5ZdfrhMnTlzYjXuBDR48WK+88opef/11zZ8/Xy+//LJ32pQpU7Rx40bvTu3bEhMT%0Ade+99+rJJ5/0eVgE3SN43x2x+LYhQ4Z4z6M4036np31Kfwrtk8Pjdrtds2bN0pAhQyTpjMPj77zz%0Ajq677jrvuS9jxozR//7v/0o6dT+9f/9+ZWdnS+reL3z22Wdn3ScHEs4el5SVlaXLL7/ce9zkyiuv%0A1NNPP61169Zp/vz5uvnmm3XVVVeprq5OUvcv+dNPP/V+/uTOKykpSWPHjtW6dev0+9//XpMmTdIV%0AV1yhiooKPfnkk1q/fr127dqljz/+WOXl5Zo7d67Wr18vSfqf//kf7/KGDh3qHRrv7OzUxx9/rISE%0AhFPWFShuvfVWXXnllXrzzTd1/PhxffHFF3ruuef0yCOP6Pjx4/rure1/9rOfadmyZRo2bJisVmuP%0A2+y7Lr/8cnV2dmrLli1nXEdUVJTGjh2rJUuWeIM4KSlJ2dnZWrdunZ5//nlNmTJFBw8eVENDg1au%0AXKnf/va3WrZsmVwul3c9S5Ys0YMPPqinn35aw4cP99YfaNu9txISEhQZGam7775b4eHhKikpOWV6%0AQUGB1qxZI6fTedpn7777btntdn300UcXq1xDcjgcev311896HkZo6De74DPtd871/6M/iIuL07Jl%0Ay7Rw4UIdPHjwjPMkJSXpk08+kcvlksfj0bZt27xhfXJbJyUladiwYfrP//xPrVu3TlOnTlVycvIZ%0A98mhoaFyu90Xp4HnqN/0tGtqak45Ieq7v/THHntMU6ZM0Z133qmCggLl5ubK5XIpJCRES5YsUWJi%0AorZs2aKsrCwNGjRIAwYMOO1M5ltvvVV/+ctfNGPGDLW3t+u2226TxWJRcnKyZsyYIbPZrEsvvVTX%0AXXedHA6H5syZI7PZrKioKN18883eP5ZbbrlFf/nLXzRt2jR1dnZq4sSJfjt2fS4ee+wxffTRRzp+%0A/Lg+//xzzZw5UyEhIbriiitO284TJ07UkiVLvCHR0zaTvhkeP/mPU1RUpAEDBuill146bR1XXHGF%0AMjMzNWPGDO8ZpD//+c/12GOPqaKiQg6HQw888IAGDx6sQ4cOKSsrS6GhoZo1a5ZMpm/+DaZMmaJf%0A/epXslqtuuyyy7zHw4NZUVGR7rrrLoWFhWny5MmSukeTFixYoPvvv/+0+UNCQrR06dKAOjknUJwc%0Ayg0NDVVXV5fmzp17yqjc2fS03+np/6M/GjZsmO655x7vsf/vSk5O1qRJkzR9+nS53W798Ic/1G23%0A3aampibvPCNGjFBqaqqmT5+ujo4OjRo1SpdeeqlGjRp12j65o6NDf//737V27Vpvz9zfeMrXOdq9%0Ae7eampr04x//WHa7XT/5yU+0efPmfnGJkFF88sknWr9+vZ555hl/lwIAFwShfY7a29s1b948HTly%0ARF1dXbr77ruVnp7u77LwD+vXr9cf/vAHPf/880pMTPR3OQBwQRDaAAAYBCeiAQBgEIQ2AAAGQWgD%0AAGAQhDYAAAZBaAMAYBCENgAABvH/AbvQsRZnvciRAAAAAElFTkSuQmCC%0A)

Ok. What is going on here?

1.  At a first sight, we could think that we should be very happy!
    However, this very high values are biased by the unbalance of the
    classes.
2.  Since some of the models assume Gaussian features centered in 0. We
    should transpose them to have better results with these models
    (e.g., SVM)

In [300]:

    1 - Y.sum()/Y.count()

Out[300]:

    0.9789579158316634

If we always predict "NO", we will reach an accuracy of almost 98%! The
classes are very unbalanced and hence we should be pay more attention to
pay and evaluate our models. First, we should take into account not
accuracy but the precision and the recall of the single classes.
Accuracy could be easily overwhelmed by the huge number of zeros.

In [301]:

    results = []
    names = []
    for name,model in models:
        kfold = KFold(n_splits=10, random_state=42)
        cv_result = cross_val_score(model,X_train,Y_train, cv = kfold,scoring = "recall")
        names.append(name)
        results.append(cv_result)
    for i in range(len(names)):
        print(names[i],results[i].mean())
        
    ax = sns.boxplot(data=results)
    ax.set_xticklabels(names)
    plt.savefig('RecInitial.png', bbox_inches='tight')
    plt.show()

    ('LRegression', 0.63800727780990929)
    ('NaiveBayesian', 0.81024455369192216)
    ('KNN', 0.17920107962213225)
    ('DTree', 0.81388167534220168)
    ('RForest', 0.7102808463466358)

![](data:image/png;base64,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%0AK1fq1KlTOnXqlJ599lk999xz2rdvn7xer6ZOnaoRI0Zo48aN2rJli8LDw3Xddddp3rx5+sc//qHV%0Aq1fLarXqyiuv1BNPPKGnnnpKnTp10sSJE5Wbm6v9+/dLku666y7dc889mjVrliIjI3Xs2DGdOHFC%0Aubm5uuaaa4zsr0GDBsnr9Wrjxo2aMmWK//nHH39cpaWlOnXqlPr166clS5ZoxYoV6tSpkz777DP1%0A69dPY8eOVXl5uaZNm6bNmzfr8ccfv2Cfna+2tlZOp1NXXHFFg9tIT0/XH//4R/Xu3Vu7d+9WUVGR%0AZsyYoblz58rhcEiS5s2bp759+2r27NkqKytTVVWVMjMzNWbMGA0dOlQ7duxQWVmZcnNzVVtbK4fD%0AoezsbKWkpOjOO+9USkqKPv30U11xxRVasWJFq+jBbNmyRRs2bNDatWvVoUMHSdLDDz+sFStWaNCg%0AQbrqqqvqrT9mzBgdOHBARUVFuv32202U3KLYbDalpaVp586d+uEPf3jB8qNHj+rXv/61OnbsqFtu%0AuUW33HLLBcee9u3bB/w/0pa43W6dOHHC/3m9mDVr1ujvf/+7rFar+vfvr6ysLK1YsUIHDhxQZWWl%0AFi9erP/85z/avn27wsLCNHLkSGVmZl70uPzMM8/o8OHDKigoUFpaWjO+04trVSE8c+ZMWa1WHT9+%0AXMnJyVqyZIkkaf78+crJyVFSUpJefvllPffcc7ruuut06tQpbdq0SRUVFbrzzjv97QwaNEhTp07V%0A7t27dfToUb300kuqrq5Wamqqbr75Zm3evFn/93//p+uvv14vvviiPB6Ptm/frvvuu0/Dhw/Xli1b%0A5HQ6/e0VFRXp6NGjKiwslMfj0aRJkzRo0CBJ0g9+8AMtWrRIhYWFKigo0KJFi5p3p50nOztbEyZM%0A0JAhQyRJTqdTsbGxWrt2rbxer0aNGqUvv/zSv/6ECRO0aNEijR07Vn/72980bty4BveZJOXl5Wn1%0A6tU6ceKEoqKi1K9fvwa3MWHCBL366qt65JFH9Morr2jatGl65plnNGjQIE2aNEmfffaZZs+erdWr%0AV6ukpESFhYWSpOLi4nrv6eOPP9bMmTPVt29fbdu2TZs3b1ZKSoo+//xzPf/88+rSpYvS09P13nvv%0AKTk5uZn2dNPYt2+fvvzyS50+fVq1tbX+5zt37qyHHnpIc+fO1V//+td6r7FYLMrNzdUvf/nLFv/+%0Am8sVV1yhQ4cONbi8vLxcr7zyiiIjI5WamnrBsSclJeWi/0fOdRragj179igjI0MnT55UeHi4UlNT%0ANXjwYJ0+fVoZGRn+9WbOnKmIiAjt2LFD+fn5slqtevDBB1VUVCRJSkxM1Lx58/Txxx/rtdde04sv%0AvihJ+vnPf64f/ehHFz0u33///crPzw+JAJZaWQifOx2dn59f7/rmkSNHtHDhQklSTU2NevbsKZvN%0A5j/oxMfHKzEx0d/O1VdfLUn68MMPdejQIf+HwuPx6NixY1qyZInWrFmjxx57TMnJyfL5fJo9e7ae%0AffZZbdiwQYmJibrjjjv87R05ckT9+/dXWFiYIiIidMMNN+jIkSOS5P82fdVVV+ntt99u4j10aXFx%0AcZozZ45mzpyplJQUtWvXTl988YUefvhhxcTEqLKyUjU1Nf71k5KSVFtbq2PHjum1117TunXrVFBQ%0AcNF9JtU/Hf3nP/9Zubm5ys7OVkVFxQXbGDFihMaNG6f77rtPX375pa655hotX75ce/bs0Y4dOyRJ%0Ap0+flt1u15w5czR//nw5nU6NHj263nu68sortWrVKrVr104ul8t/7TMuLs7/+ejSpYuqq6ubduc2%0Ag4SEBK1du1Yvv/yysrKytHr1av+y0aNHa9euXf6D1Pl69uypzMxMLVy4MOClCNSdZfv2GYXzde3a%0A1T8W4WLHnoaOK20phM+djnY4HLr33nvVtWtXSbro6egdO3bohhtu8I8f6d+/vz766CNJ9Y/Vx48f%0A19SpUyXVHRvKysoueVwOFa1ydHR6erq6dOniv+Zw9dVXa+nSpVq/fr2ysrJ02223qXfv3jp48KCk%0Auj/YZ5995n/9uQNRYmKiBg4cqPXr1+v555/XiBEj1K1bNxUWFmrhwoXasGGD3n//fR04cEAFBQV6%0A8MEHtWHDBknSP//5T397vXr18p+Krqmp0YEDB9SjR4962woVQ4cO1dVXX61XX31VVVVV+uKLL/Sn%0AP/1JDz/8sKqqqvTtqcZ/9rOfKS8vT0lJSYqNjW1wn31bly5dVFNTozfffPOi24iJidHAgQO1ePFi%0Af7AmJiZq6tSpWr9+vZYvX67Ro0frxIkTOnTokJ566in95S9/UV5enjwej387ixcv1vTp07V06VL1%0A6dPHX3+o7ffG0KNHD0VFRWnKlCmKiIjQ008/XW95dna21qxZI5fLdcFrp0yZIofDoT179jRXuS2S%0A0+nUyy+/fMmxDOHh3xxWL3bs+a7/R9qCuLg45eXlad68eTpx4sRF10lMTNS7774rj8cjn8+nkpIS%0Af/ie29eJiYlKSkrSCy+8oPXr12vcuHHq27fvRY/L4eHh8nq9zfMGv4MW2xMuLi6uN0Do23/AuXPn%0AavTo0frpT3+q7OxszZw5Ux6PR2FhYVq8eLF69uypN998U+np6erUqZPatWt3wUjdoUOH6q233tKk%0ASZNUWVmpO+64Q3a7XX379tWkSZNks9nUuXNn3XDDDXI6nZo2bZpsNptiYmJ02223+f/wt99+u956%0A6y2lpaWppqZGw4cPN3bt97uYO3eu9uzZo6qqKn3++eeaPHmywsLC1K1btwv28/Dhw7V48WL/Ab+h%0AfSZ9czr63H+CnJwctWvXTqtWrbpgG926dVNqaqomTZrkHx15//33a+7cuSosLJTT6dRvf/tbJSQk%0AqLy8XOnp6QoPD9e9994rq/Wbj/Xo0aP10EMPKTY2VldddZX/enJrl5OTozFjxshisWjkyJGS6s74%0AzJo1S7/5zW8uWD8sLExLliwJmcEqoeTcqdPw8HDV1tbqwQcfrHfm7FIaOvY09H+kLUpKSlJGRob/%0A2vm39e3bVyNGjNDEiRPl9Xp100036Y477tDhw4f96/Tr10+DBw/WxIkT5Xa7df3116tz5866/vrr%0ALzguu91uffjhh1q3bp2/52xSm72L0pEjR3T48GGNGjVKDodDd911l4qKitrMT1pagnfffVcbNmzQ%0AY489ZroUAGgSbTaEKysrNWPGDJ08eVK1tbWaMmWKxo4da7osfG3Dhg3atGmTli9frp49e5ouBwCa%0ARJsNYQAATGuVA7MAAGgJCGEAAAwhhAEAMIQQBgDAEEIYAABDCGEAAAz5f2meXXLVKKexAAAAAElF%0ATkSuQmCC%0A)

In [417]:

    from sklearn.preprocessing import StandardScaler

    X = JoinTraining[JoinTraining.columns.difference(['County','Client ID','Loan Flag'])]
    Y = JoinTraining['Loan Flag']

    rescaledX = StandardScaler().fit_transform(X)

    X_train_rescaled,X_test_rescaled,Y_train,Y_test = train_test_split(rescaledX,Y, random_state = 42, test_size = 0.2)

    models = []
    models.append(("SVM",SVC()))
    models.append(("LSVM",LinearSVC()))

    results = []
    names = []
    for name,model in models:
        kfold = KFold(n_splits=10, random_state=42)
        cv_result = cross_val_score(model,X_train_rescaled,Y_train, cv = kfold,scoring = "recall")
        names.append(name)
        results.append(cv_result)
    for i in range(len(names)):
        print(names[i],results[i].mean())
    ax = sns.boxplot(data=results)
    ax.set_xticklabels(names)
    plt.savefig('RecInitialNorm.png', bbox_inches='tight')
    plt.show()

    ('SVM', 0.9200200501253134)
    ('LSVM', 0.95990476190476193)

![](data:image/png;base64,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%0AFYtzQcE9euqpnyZ4IgCYPaZ0WRsAAMws4oyb9sgj5Z/7NQBgavhFMG7aF7/4HyoouCf2NQBgehBn%0ATAlnzAAw/XhBGAAACcALwgAAmEWIMwAAhiHOAAAYhjgDAGAY4gwAgGGIMwAAhiHOAAAYhjgDAGAY%0A4gwAgGGM+YQwAADwKc6cAQAwDHEGAMAwxBkAAMMQZwAADEOcAQAwDHEGAMAwxBkT+s1vfqONGzdq%0A/fr12rBhg7q7u1VSUqL//w68a9euqaSkRMPDwyooKNCOHTvGfI/6+nqVlJTM9OjArPPnP/9Zjz/+%0A+Jj7+vr69IMf/ECbN29WZWWlGhsbFY1G9dRTT6m1tXXMvi+++KL27Nmj5557Tvfcc4/6+/tj2z7+%0A+GMtX75cBw4cmJFjwdQQZ4zr/PnzOnbsmPbt26eXXnpJ27dv1/bt25Wbm6u33347tt+xY8f0wAMP%0AyOVyKTMzU6dOnVIkEpEkjY6O6i9/+UuiDgGY9Z599lmtX79ev/vd79TS0qILFy7o6NGjqqio0Btv%0AvDFm39dff10VFRWSpLvvvluHDx+ObXvzzTe1ePHiGZ0dN484Y1wul0sfffSRWltb1d/fr3vuuUet%0Ara2qrKzUwYMHY/u99tpr8nq9kiSHw6GvfvWr6uzslCT98Y9/1IMPPpiQ+YG5YNGiRXr99dfV1dWl%0ASCSiX/3qV1qxYoW+8pWvaHBwUB9++KEk6ezZs1q0aJFycnIkSStXrtSRI0di3+f48eP65je/mZBj%0AwI0jzhjX7bffrqamJr377rvyer36zne+o+PHj2vFihV655139Mknn+gf//iHLl26pMLCwti67373%0Au3rzzTclSYcOHdLq1asTdQjArPfUU0/py1/+sp599lk9+OCDqq2t1fDwsCRpzZo1amtrkyQdOHBA%0AVVVVsXWLFi1SWlqa/v73v6uvr0933HGHUlJSEnIMuHHEGePq6+uT0+nU7t27deLECTU2NupnP/uZ%0AQqGQVqxYoY6ODh08eFDl5eVj1t1///06d+6choaGdPnyZd15550JOgJg9vvTn/6kjRs36ve//71O%0AnDih9PR0Pf/885KkRx55RIcPH9bVq1f19ttvX3dmvGrVKv3hD39Qe3s7PyTPMsQZ4+rt7dXOnTsV%0ADoclSXl5eVqwYIHsdrsqKip06NAhdXR06OGHHx6zzmaz6etf/7rq6uq0YsWKRIwOzBmNjY2x13hk%0AZGQoLy9PycnJkqTs7GwtXbpUzz//vEpLS+VwOMasfeihh3T06FGdOnVKDzzwwIzPjpvniL8L5qtv%0Af/vb+uCDD7RmzRqlp6fLsiw9+eSTcrlccrlcCoVCWrp0qVwu13VrV69erTVr1mjnzp0JmByYvTo7%0AO/W9730vdruxsVFPP/20fD6fkpOTlZOTo7q6utj2yspKbdmyZczvl//F5XLpjjvu0F133aWkJM7F%0AZhP+KhUAAIbhRykAAAxDnAEAMAxxBgDAMMQZAADDEGcAAAxDnAEAMAxxBgDAMMQZAADD/B9xqU9a%0A8DBFzgAAAABJRU5ErkJggg==%0A)

Second, since none of these techniques fit such unbalanced use cases,
maybe we should re-train the models with a slightly modified datasets,
in which we subsample the majority class (or oversample the minority
one).

Let's start from the beginning with a brand new oversampled and
undersampled copy of the dataset. Do not forget to work just on the
training set for the resampling, to avoid overfitting.

In [303]:

    Xcol = JoinTraining.columns.difference(['County','Client ID','Loan Flag'])
    Ycol = 'Loan Flag'
    X = JoinTraining[Xcol]
    Y = JoinTraining[Ycol]
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state = 42, test_size = 0.2)

    rescaledX = X.copy()
    rescaledX[rescaledX.columns] = StandardScaler().fit_transform(rescaledX[X.columns])
    X_train_rescaled, X_test_rescaled, Y_train_rescaled, Y_test_rescaled = train_test_split(rescaledX,Y, random_state = 42, test_size = 0.2)

In [304]:

    from imblearn.over_sampling import SMOTE

    sm = SMOTE(random_state=12, ratio = 1.0)
    X_train_res, Y_train_res = sm.fit_sample(X_train, Y_train)

    X_train_res_rescaled, Y_train_res_rescaled = sm.fit_sample(X_train_rescaled, Y_train)

In [305]:

    pdtrain = X_train.copy()
    pdtrain["Loan Flag"] = Y_train
    pdtrainLoanN = pdtrain[pdtrain['Loan Flag'] == 0].sample(len(pdtrain[pdtrain['Loan Flag'] == 1]))
    pdtrainLoanY = pdtrain[pdtrain['Loan Flag'] == 1]
    pdtrain = pd.concat([pdtrainLoanN, pdtrainLoanY])
    X_train_sub = pdtrain[pdtrain.columns.difference(["Loan Flag"])]
    Y_train_sub = pdtrain["Loan Flag"]


    pdtrainRescaled = X_train_rescaled.copy()
    pdtrainRescaled["Loan Flag"] = Y_train_rescaled
    pdtrainRescaledN = pdtrainRescaled[pdtrainRescaled['Loan Flag'] == 0].sample(len(pdtrainRescaled[pdtrainRescaled['Loan Flag'] == 1]))
    pdtrainRescaledY = pdtrainRescaled[pdtrainRescaled['Loan Flag'] == 1]
    pdtrainRescaled = pd.concat([pdtrainRescaledN, pdtrainRescaledY])
    X_train_sub_rescaled = pdtrainRescaled[pdtrainRescaled.columns.difference(["Loan Flag"])]
    Y_train_sub_rescaled = pdtrainRescaled["Loan Flag"]

In [386]:

    models = []
    models.append(("SVM",SVC()))
    models.append(("LSVM",LinearSVC()))

    results = []
    names = []
    for name,model in models:
        kfold = KFold(n_splits=10, random_state=42)
        cv_result = cross_val_score(model,X_train_sub_rescaled,Y_train_sub_rescaled, cv = kfold,scoring = "recall")
        names.append(name)
        results.append(cv_result)
    print("undersampling")
    for i in range(len(names)):
        print(names[i],results[i].mean())
    ax = sns.boxplot(data=results)
    ax.set_xticklabels(names)
    plt.show()

        
        
        
    results = []
    names = []
    for name,model in models:
        kfold = KFold(n_splits=10, random_state=42)
        cv_result = cross_val_score(model,X_train_res_rescaled,Y_train_res_rescaled, cv = kfold,scoring = "recall")
        names.append(name)
        results.append(cv_result)
    print("oversampling")
    for i in range(len(names)):
        print(names[i],results[i].mean())
    ax = sns.boxplot(data=results)
    ax.set_xticklabels(names)
    plt.savefig('OverSVM.png', bbox_inches='tight')
    plt.show()

    undersampling
    ('SVM', 0.58750000000000002)
    ('LSVM', 0.58750000000000002)

![](data:image/png;base64,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%0ArkJggg==%0A)

    oversampling
    ('SVM', 0.98697520810314698)
    ('LSVM', 0.97929558937246974)

![](data:image/png;base64,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%0AMAxxBgDAMP8Dqz71vfUeMkkAAAAASUVORK5CYII=%0A)

In [387]:

    models = []
    models.append(("LRegression",LogisticRegression()))
    models.append(("NaiveBayesian",GaussianNB()))
    models.append(("DTree",DecisionTreeClassifier()))
    models.append(("RForest",RandomForestClassifier()))

    results = []
    names = []
    for name,model in models:
        kfold = KFold(n_splits=10, random_state=42)
        cv_result = cross_val_score(model,X_train_sub,Y_train_sub, cv = kfold,scoring = "recall")
        names.append(name)
        results.append(cv_result)
    print("undersampling")
    for i in range(len(names)):
        print(names[i],results[i].mean())
    ax = sns.boxplot(data=results)
    ax.set_xticklabels(names)
    plt.savefig('underOther.png', bbox_inches='tight')
    plt.show()
        
        
    results = []
    names = []
    for name,model in models:
        kfold = KFold(n_splits=10, random_state=42)
        cv_result = cross_val_score(model,X_train_res,Y_train_res, cv = kfold,scoring = "recall")
        names.append(name)
        results.append(cv_result)
    print("oversampling")
    for i in range(len(names)):
        print(names[i],results[i].mean())
    ax = sns.boxplot(data=results)
    ax.set_xticklabels(names)
    plt.savefig('overOther.png', bbox_inches='tight')
    plt.show()

    undersampling
    ('LRegression', 0.58750000000000002)
    ('NaiveBayesian', 0.58750000000000002)
    ('DTree', 0.53125)
    ('RForest', 0.578125)

![](data:image/png;base64,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%0AFy+W1+vVlClT2n1OZ511lh5++GElJCTI5/PJ5XKFP9e277shQ4aoubk5tl/cHuzQoUM655xzOt0+%0AdOjQ8LX3jo7hzo5PIhwdbS9Hezwe3XLLLRo6dKgkdfhy9MaNG3XxxReHfxZj7Nixev/99yW1fzw9%0AdOiQZs+eLen4cVZXV3fSx86eiJ+OlpSfn68hQ4aEr2ece+65WrZsmSoqKjR37lxdddVVGjVqlGpq%0AaiQd/2b4+OOPw/+/7dplWlqaLrvsMlVUVOipp55Sbm6uhg0bpurqat1zzz1as2aN3n33Xe3evVtV%0AVVW6/fbbtWbNGknS3/72t/B46enp4ZeiW1tbtXv3bo0YMaLdffV2EyZM0LnnnqsXX3xRTU1N+vTT%0AT/WHP/xBd955p5qamvTfb3n+4x//WOXl5crIyFD//v07XYv/NmTIELW2tmrr1q0d3kdSUpIuu+wy%0AlZSUhMOalpam2bNnq6KiQsuXL9eUKVNUX1+vPXv2aOXKlfrTn/6k8vJyBQKB8P2UlJTojjvu0LJl%0Ay5SZmRme/5mynqfL6/Xq+eefP+m1e5vtq4ezjo7hrn5P4PSkpKSovLxcixYtUn19fYf7pKWl6Z13%0A3lEgEFAoFNKOHTvC8W1bx7S0NGVkZOjpp59WRUWFpk2bptGjR3f42Gmz2WRZVvd8glF2xpwJb9u2%0Ard0P8vz3N8fChQs1ZcoU/fCHP1RRUZHmzZunQCCguLg4lZSUaOTIkdq6davy8/M1ePBgJSQknPAT%0AtRMmTNCbb76pmTNnyu/36+qrr5bL5dLo0aM1c+ZMOZ1OnX322br44ovl9Xp16623yul0KikpSVdd%0AdVX4m+p73/ue3nzzTbndbrW2tmrSpEm97tpvVyxcuFBvvPGGmpqa9Mknn+jGG29UXFychg0bdsL6%0ATZo0SSUlJXrkkUckdb4W0lcvR7cduKWlpUpISNDDDz98wn0MGzZM06dP18yZM8M/0Xnbbbdp4cKF%0Aqq6ultfr1S9/+UulpqaqoaFB+fn5stlsuuWWW+RwfHV4TZkyRb/61a/Uv39/nXPOOeHryehc28ub%0ANptNwWBQt99+e7tXoE6ms2O4s+8JRFdGRoYKCgrC1+X/2+jRo5Wbm6sZM2bIsixdeumluvrqq7V3%0A797wPmPGjNH48eM1Y8YMtbS06KKLLtLZZ5+tiy666ITHzpaWFu3bt0+rV68Onzn3FPwVpS7av3+/%0A9u7dq8mTJ8vj8ei6667Tli1b+NWTM8A777yjNWvW6L777jM9FQC9DBHuIr/fr8LCQh0+fFjBYFA3%0A3XSTpk6danpaiLE1a9Zo7dq1Wr58uUaOHGl6OgB6GSIMAIAh/GAWAACGEGEAAAwhwgAAGEKEAQAw%0AhAgDAGAIEQYAwJD/A2cOMTTLpVxOAAAAAElFTkSuQmCC%0A)

    oversampling
    ('LRegression', 0.97133941685631819)
    ('NaiveBayesian', 0.9769522004715977)
    ('DTree', 0.94768589736034659)
    ('RForest', 0.93955778535074452)

![](data:image/png;base64,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%0AAGCY/wdySs6DqYfeSwAAAABJRU5ErkJggg==%0A)

Some of them are very similar. In general we can just exclude some of
them and pick a group of the best to be tested with the testing set.
Let's start with a DT. Then test the random forest, Naive Bayesian,
logistic regression and a linear SVM

In [308]:

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import recall_score
    from sklearn.metrics import precision_score


    model = DecisionTreeClassifier()
    model.fit(X_train_sub,Y_train_sub)
    predictions = model.predict(X_test)

    print("Subsampling")
    print(classification_report(Y_test,predictions))


    model = DecisionTreeClassifier()
    model.fit(X_train_res,Y_train_res)
    predictions = model.predict(X_test)

    print("Oversampling")
    print(classification_report(Y_test,predictions))

    model = DecisionTreeClassifier()
    model.fit(X_train,Y_train)
    predictions = model.predict(X_test)
    print("\nWhole training set")
    print(classification_report(Y_test,predictions))

    Subsampling
                 precision    recall  f1-score   support

            0.0       1.00      0.94      0.97      1947
            1.0       0.27      0.90      0.41        49

    avg / total       0.98      0.94      0.95      1996

    Oversampling
                 precision    recall  f1-score   support

            0.0       1.00      0.99      0.99      1947
            1.0       0.72      0.94      0.81        49

    avg / total       0.99      0.99      0.99      1996


    Whole training set
                 precision    recall  f1-score   support

            0.0       1.00      1.00      1.00      1947
            1.0       0.86      0.88      0.87        49

    avg / total       0.99      0.99      0.99      1996

In [309]:

    model = RandomForestClassifier()
    model.fit(X_train_sub,Y_train_sub)
    predictions = model.predict(X_test)

    print("Subsampling")
    print(classification_report(Y_test,predictions))


    model = RandomForestClassifier()
    model.fit(X_train_res,Y_train_res)
    predictions = model.predict(X_test)

    print("Oversampling")
    print(classification_report(Y_test,predictions))

    model = RandomForestClassifier()
    model.fit(X_train,Y_train)
    predictions = model.predict(X_test)
    print("\nWhole training set")
    print(classification_report(Y_test,predictions))

    Subsampling
                 precision    recall  f1-score   support

            0.0       1.00      0.95      0.98      1947
            1.0       0.34      0.96      0.50        49

    avg / total       0.98      0.95      0.96      1996

    Oversampling
                 precision    recall  f1-score   support

            0.0       1.00      1.00      1.00      1947
            1.0       0.87      0.84      0.85        49

    avg / total       0.99      0.99      0.99      1996


    Whole training set
                 precision    recall  f1-score   support

            0.0       0.99      1.00      1.00      1947
            1.0       0.97      0.73      0.84        49

    avg / total       0.99      0.99      0.99      1996

In [310]:

    DT = LinearSVC()
    model.fit(X_train_sub_rescaled,Y_train_sub_rescaled)
    predictions = model.predict(X_test_rescaled)

    print("Subsampling")
    print(classification_report(Y_test_rescaled,predictions))

    DT = LinearSVC()
    model.fit(X_train_res_rescaled,Y_train_res_rescaled)
    predictions = model.predict(X_test_rescaled)

    print("Oversampling")
    print(classification_report(Y_test_rescaled,predictions))

    DT = LinearSVC()
    model.fit(X_train_rescaled,Y_train_rescaled)
    predictions = model.predict(X_test_rescaled)
    print("\nWhole training set")
    print(classification_report(Y_test_rescaled,predictions))

    Subsampling
                 precision    recall  f1-score   support

            0.0       1.00      0.97      0.99      1947
            1.0       0.48      0.96      0.64        49

    avg / total       0.99      0.97      0.98      1996

    Oversampling
                 precision    recall  f1-score   support

            0.0       1.00      1.00      1.00      1947
            1.0       0.87      0.82      0.84        49

    avg / total       0.99      0.99      0.99      1996


    Whole training set
                 precision    recall  f1-score   support

            0.0       0.99      1.00      1.00      1947
            1.0       0.92      0.71      0.80        49

    avg / total       0.99      0.99      0.99      1996

In [311]:

    model = GaussianNB()
    model.fit(X_train_sub,Y_train_sub)
    predictions = model.predict(X_test)

    print("Subsampling")
    print(classification_report(Y_test,predictions))


    model = GaussianNB()
    model.fit(X_train_res,Y_train_res)
    predictions = model.predict(X_test)

    print("Oversampling")
    print(classification_report(Y_test,predictions))

    model = GaussianNB()
    model.fit(X_train,Y_train)
    predictions = model.predict(X_test)
    print("\nWhole training set")
    print(classification_report(Y_test,predictions))

    Subsampling
                 precision    recall  f1-score   support

            0.0       1.00      0.89      0.94      1947
            1.0       0.18      0.96      0.30        49

    avg / total       0.98      0.89      0.92      1996

    Oversampling
                 precision    recall  f1-score   support

            0.0       1.00      0.92      0.96      1947
            1.0       0.23      0.96      0.37        49

    avg / total       0.98      0.92      0.94      1996


    Whole training set
                 precision    recall  f1-score   support

            0.0       0.99      1.00      0.99      1947
            1.0       0.80      0.71      0.75        49

    avg / total       0.99      0.99      0.99      1996

In [312]:

    model = LogisticRegression()
    model.fit(X_train_sub,Y_train_sub)
    predictions = model.predict(X_test)

    print("Subsampling")
    print(classification_report(Y_test,predictions))

    model = LogisticRegression()
    model.fit(X_train_res,Y_train_res)
    predictions = model.predict(X_test)

    print("Oversampling")
    print(classification_report(Y_test,predictions))

    model = LogisticRegression()
    model.fit(X_train,Y_train)
    predictions = model.predict(X_test)
    print("\nWhole training set")
    print(classification_report(Y_test,predictions))

    Subsampling
                 precision    recall  f1-score   support

            0.0       1.00      0.92      0.96      1947
            1.0       0.24      0.96      0.38        49

    avg / total       0.98      0.92      0.95      1996

    Oversampling
                 precision    recall  f1-score   support

            0.0       1.00      0.96      0.98      1947
            1.0       0.37      0.96      0.54        49

    avg / total       0.98      0.96      0.97      1996


    Whole training set
                 precision    recall  f1-score   support

            0.0       0.99      1.00      1.00      1947
            1.0       0.95      0.78      0.85        49

    avg / total       0.99      0.99      0.99      1996

The random forest and Logistic regression seem the most promising.

Now we have to decide. Do we want something generally accurate or we
have some specific requirements? I really want to give more attentions
to the customers with a loan, wanting to be able to predict all of them
even at the cost of being to much generous with this class assignment.

If we were interested to general accuracy, we would have used not the
undersampled training set. In this way, we could have detected just the
majority of the true positives, however with much less false positives.

The model trained with the subsampled dataset is able to detect all the
ones (some of them belong to the training set, but still in the real
test it seems very covering). It predicts 1 maybe too many times but
with such so unbalanced training set, this is not bad!

The choicse depends on the business. In this specific case I prefer not
to miss any true positive. Go ahead with the subsampling. Try to tune
the forest.

In [404]:

    sample_leaf_options = [1,5,10,25,50,100]
    for leaf_size in sample_leaf_options :
        model = RandomForestClassifier(n_estimators = 100,  n_jobs = -1,random_state =50,  max_features = "auto", min_samples_leaf = leaf_size)
        model.fit(X_train_sub,Y_train_sub)
        predictions = model.predict(X_test)
        print leaf_size
        print "recall : ", recall_score(Y_test,predictions)
        print "precision : ", precision_score(Y_test,predictions)
                   

    1
    recall :  0.959183673469
    precision :  0.284848484848
    5
    recall :  0.959183673469
    precision :  0.22380952381
    10
    recall :  0.959183673469
    precision :  0.200854700855
    25
    recall :  0.959183673469
    precision :  0.18875502008
    50
    recall :  0.959183673469
    precision :  0.164912280702
    100
    recall :  0.877551020408
    precision :  0.0957683741648

In [405]:

    estimators = [1,5,10,25,50,100,150,200,500]
    for estimator in estimators :
        model = RandomForestClassifier(n_estimators = estimator, n_jobs = -1,random_state =50,  max_features = "auto", min_samples_leaf = 1)
        model.fit(X_train_sub,Y_train_sub)
        predictions = model.predict(X_test)
        print estimator
        print "recall : ", recall_score(Y_test,predictions)
        print "precision : ", precision_score(Y_test,predictions)

    1
    recall :  0.857142857143
    precision :  0.121037463977
    5
    recall :  0.938775510204
    precision :  0.270588235294
    10
    recall :  0.918367346939
    precision :  0.31914893617
    25
    recall :  0.959183673469
    precision :  0.307189542484
    50
    recall :  0.959183673469
    precision :  0.278106508876
    100
    recall :  0.959183673469
    precision :  0.284848484848
    150
    recall :  0.959183673469
    precision :  0.315436241611
    200
    recall :  0.959183673469
    precision :  0.311258278146
    500
    recall :  0.959183673469
    precision :  0.335714285714

In [406]:

    features = np.arange(1,10)
    for n in features :
        model = RandomForestClassifier(n_estimators = 200, n_jobs = -1,random_state =50,  max_features = n, min_samples_leaf = 1)
        model.fit(X_train_sub,Y_train_sub)
        predictions = model.predict(X_test)
        print n
        print "recall : ", recall_score(Y_test,predictions)
        print "precision : ", precision_score(Y_test,predictions)

    1
    recall :  0.959183673469
    precision :  0.255434782609
    2
    recall :  0.959183673469
    precision :  0.271676300578
    3
    recall :  0.959183673469
    precision :  0.311258278146
    4
    recall :  0.959183673469
    precision :  0.324137931034
    5
    recall :  0.959183673469
    precision :  0.350746268657
    6
    recall :  0.959183673469
    precision :  0.338129496403
    7
    recall :  0.959183673469
    precision :  0.345588235294
    8
    recall :  0.959183673469
    precision :  0.343065693431
    9
    recall :  0.959183673469
    precision :  0.311258278146

In [407]:

    model = RandomForestClassifier(n_estimators = 150, n_jobs = -1, random_state =50,  max_features = 7, min_samples_leaf = 1)
    model.fit(X_train_sub,Y_train_sub)
    predictions = model.predict(X_test)
    print "recall : ", recall_score(Y_test,predictions)
    print "precision : ", precision_score(Y_test,predictions)

    recall :  0.959183673469
    precision :  0.338129496403

Let's try with Logistic Regression'

In [408]:

    param=[0.001, 0.01, 0.1, 1, 10, 100, 1000]

    for c in param :
        model = LogisticRegression(C=c, intercept_scaling=1, dual=False, fit_intercept=True,
              penalty='l2', tol=0.0001)
        model.fit(X_train_sub,Y_train_sub)
        predictions = model.predict(X_test)
        print n
        print "recall : ", recall_score(Y_test,predictions)
        print "precision : ", precision_score(Y_test,predictions)
        

    param=[0.001, 0.01, 0.1, 1, 10, 100, 1000]

    9
    recall :  0.938775510204
    precision :  0.129213483146
    9
    recall :  0.938775510204
    precision :  0.145110410095
    9
    recall :  0.938775510204
    precision :  0.175572519084
    9
    recall :  0.959183673469
    precision :  0.239795918367
    9
    recall :  0.959183673469
    precision :  0.307189542484
    9
    recall :  0.959183673469
    precision :  0.321917808219
    9
    recall :  0.959183673469
    precision :  0.315436241611

In the end we decided to use the Random Forest.

One last try. Since we have excluded the information about the county,
which could be usefull, let's try to include it. Of course, since it is
categorical, we should pay attention (even if we transform it into
integers, we do not want that our algorithm think that they could be
ordererd).

So we create a new (boolean) feature for each of the possible values.

In [409]:

    from sklearn.preprocessing import LabelEncoder, OneHotEncoder

    one_hot = pd.get_dummies(JoinTest['County'])
    JoinTrainingCat = JoinTraining.join(one_hot)

    Xcolcat = JoinTraining.columns.difference(['County','Client ID','Loan Flag'])
    Ycol = 'Loan Flag'
    Xcat = JoinTraining[Xcolcat]
    Y = JoinTraining[Ycol]
    X_traincat, X_testcat, Y_traincat, Y_testcat = train_test_split(Xcat,Y, random_state = 42, test_size = 0.2)

    sm = SMOTE(random_state=12, ratio = 1.0)
    X_train_rescat, Y_train_rescat = sm.fit_sample(X_traincat, Y_traincat)

    pdtraincat = X_traincat.copy()
    pdtraincat["Loan Flag"] = Y_traincat
    pdtraincatLoanN = pdtraincat[pdtraincat['Loan Flag'] == 0].sample(len(pdtraincat[pdtraincat['Loan Flag'] == 1]))
    pdtraincatLoanY = pdtraincat[pdtraincat['Loan Flag'] == 1]
    pdtraincat = pd.concat([pdtraincatLoanN, pdtraincatLoanY])
    X_train_subcat = pdtraincat[pdtraincat.columns.difference(["Loan Flag"])]
    Y_train_subcat = pdtraincat["Loan Flag"]

    model = RandomForestClassifier(n_estimators = 150, n_jobs = -1, random_state =50,  max_features = 7, min_samples_leaf = 1)
    model.fit(X_train_subcat,Y_train_subcat)
    predictions = model.predict(X_test)
    print "recall : ", recall_score(Y_test,predictions)
    print "precision : ", precision_score(Y_test,predictions)

    recall :  0.959183673469
    precision :  0.484536082474

Ok, it is not good as the one without these features. roll-up the
transformation. Let's now use the label probabilities of the classifier.
Apply it to the whole dataset. The recall and precision values are not
valid because all the 1s of the training set are the same of the testing
set.

In [410]:

    model = RandomForestClassifier(n_estimators = 150, n_jobs = -1, random_state =50,  max_features = 7, min_samples_leaf = 1)
    model.fit(X_train_sub,Y_train_sub)
    predictions = model.predict(JoinTraining[JoinTraining.columns.difference(['County','Client ID','Loan Flag'])])
    prob = model.predict_proba(JoinTraining[JoinTraining.columns.difference(['County','Client ID','Loan Flag'])])
    print "recall : ", recall_score(JoinTraining["Loan Flag"],predictions)
    print "precision : ", precision_score(JoinTraining["Loan Flag"],predictions)

    recall :  0.990476190476
    precision :  0.302765647744

In [411]:

    model_Final = model

In [412]:

    JoinTraining["loan_predicted"] = predictions
    JoinTraining["prob"] = prob[:,1]

As we can see, we do not miss a loan, but we are kind of generous with
its assignment.

In [413]:

    JoinTraining.ix[JoinTraining["Loan Flag"]==1, ["Loan Flag","loan_predicted"]].head(10)

Out[413]:

Loan Flag

loan\_predicted

25

1.0

1.0

76

1.0

1.0

95

1.0

1.0

131

1.0

1.0

176

1.0

1.0

249

1.0

1.0

511

1.0

1.0

538

1.0

1.0

577

1.0

1.0

581

1.0

1.0

In [414]:

    JoinTraining.ix[JoinTraining["loan_predicted"]==1, ["Loan Flag","loan_predicted"]].head(10)

Out[414]:

Loan Flag

loan\_predicted

25

1.0

1.0

69

0.0

1.0

76

1.0

1.0

92

0.0

1.0

95

1.0

1.0

98

0.0

1.0

114

0.0

1.0

122

0.0

1.0

131

1.0

1.0

132

0.0

1.0

In [415]:

    Right = JoinTraining.ix[JoinTraining["loan_predicted"] == JoinTraining["Loan Flag"], ["Loan Flag","loan_predicted","prob"]]
    Mistakes = JoinTraining.ix[JoinTraining["loan_predicted"] != JoinTraining["Loan Flag"], ["Loan Flag","loan_predicted","prob"]]

In [326]:

    bins = np.linspace(0.5, 1, 41)

    plt.hist(Mistakes.prob, bins, alpha=0.5, normed = True, label='false positives')


    plt.hist(Right.prob, bins, alpha=0.5, normed = True, label='true positives')
    plt.xlabel("Confidence of our model")
    plt.ylabel("Predictions")
    plt.legend()
    plt.savefig('Mistakes.png', bbox_inches='tight')
    plt.show()

![](data:image/png;base64,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%0AohXs8LvYocYbjaf/TZZtDPL481hfj+3p34On5/OU55eAS/53pW+Uy9m+5te7xzdt2qSsrCy99957%0AGj58uKZOnerP4QEAsDW/zrS/+OILNWvWTJJ07733au/evf4cHgBuCJ7OYsuW5RNSu/Hrf7HLly+r%0AYsWKeT8HBATI5XLJ6bx+GRERlXxeQ0K3P/r8nL5lhwUj7FCjp0rT71LS2aHXvq7RDr8zfMGKvLoe%0Av14er1ixotLT0/N+zs3NLTCwAQBAfn4N7YYNG2rr1q2SpK+++kp33nmnP4cHAMDWHMYY46/Bfr57%0A/JtvvpExRpMnT1atWrX8NTwAALbm19AGAAD/OzYMAQDAJghtAABsolSGdm5ursaPH69evXopJiZG%0AJ06cyPf8O++8o44dOyomJkYxMTE6erRk7y9cHArr4Z49e9SnTx/17t1bzz77rDIzfbsyVWngrofJ%0Aycl5f38xMTFq1KiRli9fXozVllyF/S2uXbtW0dHR6tGjh5YtW1ZMVZZshfVwzZo16ty5s/r06aO/%0A/e1vxVSlPezevVsxMTHXPL5582b16NFDvXr10sqVK60rwJRCGzduNKNGjTLGGLNr1y4zZMiQfM8P%0AHz7cfP3118VRmm2462Fubq7p0qWLOX78uDHGmJUrV5ojR44US50lWWF/hz/78ssvTUxMjHG5XP4s%0AzzYK6+MDDzxgUlNTTWZmpmnbtq25cOFCcZRZornr4Q8//GBatWplUlNTTU5OjomJiTEnT54srlJL%0AtAULFphOnTqZRx99NN/jWVlZeX97mZmZpnv37iY5OdmSGkrlTLuwldf27dunBQsWqHfv3po/f35x%0AlFjiuevhsWPHFBoaqnfeeUf9+vXThQsXVLNm8WxpWJJ5sgKgMUYJCQmaOHGiAgIC/F2iLRTWxzp1%0A6igtLU1ZWVkyxsjhcBRHmSWaux6eOnVKderUUWhoqMqUKaN77rlHu3fvLq5SS7Tq1atr9uzZ1zx+%0A5MgRVa9eXTfddJMCAwN133336bPPfLvu/M9KZWgXtPLazzp27KiJEydq0aJF+uKLL/Txxx8XR5kl%0AmrsepqamateuXerXr58WLlyonTt3aseOHcVVaolV2N+h9NMltdq1a/Omx43C+li7dm316NFDHTt2%0AVMuWLRUSElIcZZZo7noYGRmpw4cPKyUlRVeuXNGOHTuUkZFRXKWWaO3bt7/ugmCXL19WpUq/rIhW%0A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%0AzJ979uutSa+u+2q//rz813uNX6/XxhgFBwfn6+3Vr8vNzdWiRYsUGhoqSTp79qzCw8O1adOmAn8v%0AoKRhpg14qFGjRqpcubJee+21vAD75JNPtHr1at1xxx2KiorS+vXr9eOPP8rlcmnVqlWKiooq8jhN%0AmzbVunXrlJubq9OnT+vLL7+U9NPn5//4xz+Unp4ul8uloUOHauPGjQWe55577tHu3buVnJwsSZo8%0AebISExMVFRWl5cuXS5LOnTunLl266MyZM/leGxUVpTVr1ignJ0dXrlzRBx98oCZNmhQ4VuPGjbV5%0A82ZduHBB0k93xf98fFhYmA4dOiRjjDZv3lzkflxPzZo1deHCBe3Zs0eS9OGHH6pKlSoKCwtTlSpV%0AtGXLFknKd4d7VFSUli1bJkk6fPiwunTpoitXrvikHsBfmGkDHnI4HJozZ46mTJmiTp06yel0Kiws%0ATAsWLFB4eLhatWqlAwcOqEePHnK5XGrWrJn69eun77//vkjj9OnTR4cOHVKHDh1022235W2j2Lp1%0Aax08eFA9e/ZUTk6OmjVrpujoaJ0+ffq657nllls0duxYDRo0SLm5ubr33nvVvXt3XblyRRMnTlSn%0ATp2Uk5OjkSNHqnr16vle26tXLx0/flxdu3ZVdna2unTponbt2hVYc926dfXkk08qJiZG2dnZqlev%0AnuLj4yVJw4cP15AhQxQeHq777rsv3411/6vAwEDNmDFDCQkJunLlim666SbNmDFD0k8fA4wZM0Z/%0A/etfde+99+a9Ji4uTuPHj1fnzp0lSS+//LIqVqzodS2AP7HLFwAANsHlcQAAbILQBgDAJghtAABs%0AgtAGAMAmCG0AAGyC0AYAwCYIbQAAbILQBgDAJv4fMrLBx3oEHwQAAAAASUVORK5CYII=%0A)

Try to select confidence thresholds in order to divide users in groups
with a certain likelihood to ask for a loan. Here you could obtain
different values because it depends on the sample at each run of the
notebook. The loan expectations with which the thresholds have been
extracted were 90%,70%,50%,30%, \<10% (the model with these values has
been used to obtain the final data).

In [327]:

    x = 0.95
    y = 1
    sum(JoinTraining.ix[(JoinTraining["prob"] <= y) & (JoinTraining["prob"] > x), "Loan Flag"]) / len(JoinTraining[(JoinTraining["prob"] <= y) & (JoinTraining["prob"] > x)])

Out[327]:

    0.85483870967741937

In [328]:

    x = 0.90
    y = 0.95
    sum(JoinTraining.ix[(JoinTraining["prob"] <= y) & (JoinTraining["prob"] > x), "Loan Flag"]) / len(JoinTraining[(JoinTraining["prob"] <= y) & (JoinTraining["prob"] > x)])

Out[328]:

    0.70454545454545459

In [329]:

    x = 0.86
    y =0.90
    sum(JoinTraining.ix[(JoinTraining["prob"] <= y) & (JoinTraining["prob"] > x), "Loan Flag"]) / len(JoinTraining[(JoinTraining["prob"] <= y) & (JoinTraining["prob"] > x)])

Out[329]:

    0.44444444444444442

In [330]:

    x = 0.82
    y = 0.86
    sum(JoinTraining.ix[(JoinTraining["prob"] <= y) & (JoinTraining["prob"] > x), "Loan Flag"]) / len(JoinTraining[(JoinTraining["prob"] <= y) & (JoinTraining["prob"] > x)])

Out[330]:

    0.083333333333333329

In [331]:

    x = 0
    y = 0.82
    sum(JoinTraining.ix[(JoinTraining["prob"] <= y) & (JoinTraining["prob"] > x), "Loan Flag"]) / len(JoinTraining[(JoinTraining["prob"] <= y) & (JoinTraining["prob"] > x)])

Out[331]:

    0.0015342129487572874

The thresholds will delivers approximately likelihoods of 90% 70& 50%
30% and less than 10%

In [332]:

    def classLike(x):
        if x<0.82 :
            return 1
        elif x>0.82 and x <= 0.86:
            return 2
        elif x>0.86 and x <= 0.90:
            return 3
        elif x>0.90 and x <= 0.95:
            return 4
        else:
            return 5  

In [333]:

    bins = np.linspace(0, 1, 100)
    plt.hist(JoinTraining.prob, bins, alpha=0.5, normed = True)
    plt.axvline(0.82,color='b')
    plt.axvline(0.86,color='b')
    plt.axvline(0.90,color='b')
    plt.axvline(0.95,color='b')
    plt.show()


    bins = np.linspace(0.5, 1, 41)
    plt.hist(Mistakes.prob, bins, alpha=0.5, label='false positives')
    plt.hist(Right.prob, bins, alpha=0.5, label='true positives')
    plt.legend()
    plt.axvline(0.82,color='b')
    plt.axvline(0.86,color='b')
    plt.axvline(0.90,color='b')
    plt.axvline(0.95,color='b')
    plt.savefig('thresholds.png', bbox_inches='tight')
    plt.show()

![](data:image/png;base64,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%0A)

![](data:image/png;base64,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%0ABiCQAQAwwP8DVyz+xlkovKwAAAAASUVORK5CYII=%0A)

In [334]:

    JoinTraining["Loan Likelihood"] = map (lambda x: classLike(x), JoinTraining["prob"])

Now we have the class feature for each customer.

In [335]:

    JoinTraining.head()

Out[335]:

Age

Client ID

County

Density (/ km2)

Gender

Held Loan previously

Income Group

Last TXN Amount

Loan Flag

Num Transactions

ProductsAmount

TxnAmount

loan\_predicted

prob

Loan Likelihood

0

36.0

1.0

Cork

69.0

1.0

1.0

2

0.00

0.0

0.0

4.0

58.0

0.0

0.000000

1

1

24.0

9.0

Cork

69.0

0.0

0.0

4

45.00

0.0

5.0

4.0

22.0

0.0

0.033333

1

2

57.0

24.0

Cork

69.0

1.0

0.0

2

890.00

0.0

11.0

2.0

0.0

0.0

0.313333

1

3

21.0

37.0

Cork

69.0

1.0

1.0

2

247.65

0.0

4.0

1.0

28.0

0.0

0.000000

1

4

60.0

38.0

Cork

69.0

0.0

0.0

5

220.27

0.0

9.0

4.0

0.0

0.0

0.013333

1

Let's prepare now the testing set. In this boring phase I will repeat
the same analysis and transformation I did for the training sets. At the
end I will use the model previously obtained to generate the predicted
loading files. My Priority is to maximize the recall of the Loans. My
model is not that prudent to assign 1, but it tends to hit all of them.
The price is that it tends to assign 1 even when it should not, but this
is a good tradeoff since the class were unbalanced. If I had more more,
I would have tried oversampling instead of subsampling.

In [336]:

    DemographicsTest = pd.read_csv('Test - Demographics.csv', header=0, encoding='ISO-8859-15')

In [337]:

    DemographicsTest.head()

Out[337]:

Client ID

Age

Gender 1: Female, 2: Male

County

Income Group

0

10001

59

1

Cork

10001 - 40000

1

10002

27

1

Kerry

10001 - 40000

2

10003

58

0

Louth

10001 - 40000

3

10004

45

1

Dublin

60001 - 100000

4

10005

21

0

Dublin

40001 - 60000

In [338]:

    DemographicsTest.describe()

Out[338]:

Client ID

Age

Gender 1: Female, 2: Male

count

2000.000000

2000.000000

2000.000000

mean

11000.500000

48.386500

0.509000

std

577.494589

17.710193

0.500044

min

10001.000000

20.000000

0.000000

25%

10500.750000

33.000000

0.000000

50%

11000.500000

48.000000

1.000000

75%

11500.250000

63.000000

1.000000

max

12000.000000

80.000000

1.000000

In [339]:

    DemographicsTest.columns

Out[339]:

    Index([u'Client ID', u'Age', u'Gender \n1: Female, 2: Male', u'County',
           u'Income Group'],
          dtype='object')

In [340]:

    DemographicsTest = DemographicsTest.rename(columns={'Gender \n1: Female, 2: Male' : "Gender"})

In [341]:

    DemographicsTest.groupby("Client ID").filter(lambda x: len(x) > 1)

Out[341]:

Client ID

Age

Gender

County

Income Group

In [342]:

    DemographicsTest.Gender.unique()

Out[342]:

    array([1, 0], dtype=int64)

In [343]:

    DemographicsTest['Income Group'].unique()

Out[343]:

    array([u'10001 - 40000', u'60001 - 100000', u'40001 - 60000', u'0 - 10000',
           u'100000+'], dtype=object)

In [344]:

    DemographicsTest.ix[DemographicsTest['County'].str.contains("Dub"),"County"] = "Dublin"
    DemographicsTest.ix[DemographicsTest['County'].str.contains("dub"),"County"] = "Dublin"
    DemographicsTest.ix[DemographicsTest['County'].str.contains("blin"),"County"] = "Dublin"
    DemographicsTest.ix[DemographicsTest['County'].str.contains("cork"),"County"] = "Cork"
    DemographicsTest.ix[DemographicsTest['County'].str.contains("Cork"),"County"] = "Cork"
    DemographicsTest.ix[DemographicsTest['County'].str.contains("ongford"),"County"] = "Longford"

In [345]:

    CountyTest = County.copy()

In [346]:

    PreviousLoanTest = pd.read_csv('Test - Previous Loan Holdings.csv', header=0)
    PreviousLoanTest.head()

Out[346]:

Client ID

Held Loan previously

Unnamed: 2

Unnamed: 3

Unnamed: 4

Unnamed: 5

Unnamed: 6

Unnamed: 7

Unnamed: 8

Unnamed: 9

Unnamed: 10

Unnamed: 11

Unnamed: 12

Unnamed: 13

0

10001

0

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

1

10002

0

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

2

10003

0

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

3

10004

0

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

4

10005

0

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

NaN

In [347]:

    PreviousLoanTest.columns

Out[347]:

    Index([u'Client ID', u'Held Loan previously', u'Unnamed: 2', u'Unnamed: 3',
           u'Unnamed: 4', u'Unnamed: 5', u'Unnamed: 6', u'Unnamed: 7',
           u'Unnamed: 8', u'Unnamed: 9', u'Unnamed: 10', u'Unnamed: 11',
           u'Unnamed: 12', u'Unnamed: 13'],
          dtype='object')

In [348]:

    PreviousLoanTest = PreviousLoanTest[PreviousLoanTest.columns[0:2]]
    PreviousLoanTest = PreviousLoanTest[0:2000]
    PreviousLoanTest.head()

Out[348]:

Client ID

Held Loan previously

0

10001

0

1

10002

0

2

10003

0

3

10004

0

4

10005

0

In [349]:

    PreviousLoanTest.groupby("Client ID").filter(lambda x: len(x) > 1)

Out[349]:

Client ID

Held Loan previously

In [350]:

    PreviousLoanTest.describe()

Out[350]:

Client ID

Held Loan previously

count

2000.000000

2000.000000

mean

11000.500000

0.243500

std

577.494589

0.429302

min

10001.000000

0.000000

25%

10500.750000

0.000000

50%

11000.500000

0.000000

75%

11500.250000

0.000000

max

12000.000000

1.000000

In [351]:

    BankProductsTest = pd.read_csv('Test - Product Held in Bank.csv', header=[0,1])
    BankProductsTest.head()

Out[351]:

Unnamed: 0\_level\_0

Unnamed: 1\_level\_0

Unnamed: 2\_level\_0

Unnamed: 3\_level\_0

Unnamed: 4\_level\_0

Unnamed: 5\_level\_0

0

Client ID

\# Products in bank

Unnamed: 2\_level\_1

Unnamed: 3\_level\_1

Unnamed: 4\_level\_1

Unnamed: 5\_level\_1

Unnamed: 6\_level\_1

0

10001

4

NaN

NaN

NaN

NaN

NaN

1

10002

4

NaN

NaN

NaN

NaN

NaN

2

10003

2

NaN

NaN

NaN

NaN

NaN

3

10004

2

NaN

NaN

NaN

NaN

NaN

4

10005

1

NaN

NaN

NaN

NaN

NaN

In [352]:

    BankProductsTest.columns = BankProductsTest.columns.droplevel()
    BankProductsTest = BankProductsTest[BankProductsTest.columns[0:2]]

In [353]:

    BankProductsTest.head()

Out[353]:

Client ID

\# Products in bank

0

10001

4

1

10002

4

2

10003

2

3

10004

2

4

10005

1

Change the name, I want to be consistent with the other file

In [354]:

    BankProductsTest = BankProductsTest.rename(columns={'# Products in bank' : "ProductsAmount"})
    BankProductsTest.head() #the files are the same as the training one.

Out[354]:

Client ID

ProductsAmount

0

10001

4

1

10002

4

2

10003

2

3

10004

2

4

10005

1

In [355]:

    BankProductsTest.describe()

Out[355]:

Client ID

ProductsAmount

count

2000.000000

2000.000000

mean

11000.500000

3.003000

std

577.494589

1.409604

min

10001.000000

1.000000

25%

10500.750000

2.000000

50%

11000.500000

3.000000

75%

11500.250000

4.000000

max

12000.000000

5.000000

In [356]:

    BankProductsTest.ProductsAmount.value_counts()

Out[356]:

    3    404
    2    403
    4    401
    5    398
    1    394
    Name: ProductsAmount, dtype: int64

In [357]:

    TransactionsOutTest = pd.read_csv('Model Build - Transactions out of Current Account.csv', header=0)
    TransactionsOutTest.head()

Out[357]:

Client

Num Transactions

Last TXN Amount

Merchant Code

Last Transaction Narrative

0

10706

4

879.40

3700

0002 - MOTEL 6 SANTA BARBARACA

1

11992

2

898.46

3700

0025 - MOTEL 6 GILROY CA

2

11284

4

216.43

4723

0340708011347459044271351BERLIN

3

10412

27

330.20

5021

121 OFFICE FURNITURE CUMBERNAULD

4

11183

9

1010.39

6300

123 MONEY LTD DUBLIN 18

In [358]:

    TransactionsOutTest.describe()

Out[358]:

Client

Num Transactions

Last TXN Amount

Merchant Code

count

2000.000000

2000.000000

2000.000000

2000.000000

mean

11000.500000

16.339000

594.084485

4806.407000

std

577.494589

24.264263

378.781884

1784.107396

min

10001.000000

0.000000

1.250000

742.000000

25%

10500.750000

4.000000

267.935000

3509.000000

50%

11000.500000

7.000000

588.195000

3787.500000

75%

11500.250000

10.000000

907.842500

5945.000000

max

12000.000000

100.000000

5200.240000

9405.000000

In [359]:

    TxnAmountTest = pd.read_csv('Test - TXN Amount.csv', encoding='ISO-8859-15', header=[0,1])
    #TxnAmount = pd.read_csv('Model Build - TXN Amount.csv', header=None)
    TxnAmountTest.head()

Out[359]:

Unnamed: 0\_level\_0

Unnamed: 1\_level\_0

Unnamed: 2\_level\_0

Unnamed: 3\_level\_0

Unnamed: 4\_level\_0

Unnamed: 5\_level\_0

Unnamed: 6\_level\_0

0

Client ID

Average amount of CA transaction

Unnamed: 2\_level\_1

Unnamed: 3\_level\_1

Unnamed: 4\_level\_1

Unnamed: 5\_level\_1

Unnamed: 6\_level\_1

Unnamed: 7\_level\_1

0

10001

22

NaN

NaN

NaN

NaN

NaN

NaN

1

10002

11

NaN

NaN

NaN

NaN

NaN

NaN

2

10003

9

NaN

NaN

NaN

NaN

NaN

NaN

3

10004

34

NaN

NaN

NaN

NaN

NaN

NaN

4

10005

38

NaN

NaN

NaN

NaN

NaN

NaN

In [360]:

    TxnAmountTest.columns = TxnAmountTest.columns.droplevel()
    TxnAmountTest = TxnAmountTest[TxnAmountTest.columns[0:2]]
    TxnAmountTest.head()

Out[360]:

Client ID

Average amount of CA transaction

0

10001

22

1

10002

11

2

10003

9

3

10004

34

4

10005

38

In [361]:

    TxnAmountTest = TxnAmountTest.rename(columns={'Average amount of CA transaction' : "TxnAmount"})
    TxnAmountTest = TxnAmountTest.rename(columns={'Client ID' : "Client"}) #don't like it but I did this before, 
                                                  #in a hurry and I do want to recycle code
        
    TxnAmountTest.head()

Out[361]:

Client

TxnAmount

0

10001

22

1

10002

11

2

10003

9

3

10004

34

4

10005

38

In [362]:

    TxnAmountTest.describe()

Out[362]:

Client

TxnAmount

count

2000.000000

2000.000000

mean

11000.500000

1239.594000

std

577.494589

965.085733

min

10001.000000

0.000000

25%

10500.750000

318.750000

50%

11000.500000

1181.500000

75%

11500.250000

2033.500000

max

12000.000000

9470.000000

In [363]:

    ids = np.arange(10001,12000)

In [364]:

    TargetTest = pd.DataFrame(columns=["Client ID","Loan Flag"])
    TargetTest["Client ID"] = ids
    TargetTest.head()

Out[364]:

Client ID

Loan Flag

0

10001

NaN

1

10002

NaN

2

10003

NaN

3

10004

NaN

4

10005

NaN

In [365]:

    JoinTest2 = pd.merge(DemographicsTest,CountyTest, how = 'outer', left_on='County', right_on='County')
    JoinTest2 = pd.merge(JoinTest2,PreviousLoanTest, how = 'outer', left_on='Client ID', right_on='Client ID')
    JoinTest2 = pd.merge(JoinTest2,BankProductsTest, how = 'outer', left_on='Client ID', right_on='Client ID')
    JoinTest2 = pd.merge(JoinTest2,TransactionsOutTest, how = 'outer', left_on='Client ID', right_on='Client')
    JoinTest2 = pd.merge(JoinTest2,TxnAmountTest, how = 'outer', left_on='Client ID', right_on='Client')
    JoinTest2 = pd.merge(JoinTest2,TargetTest, how = 'outer', left_on='Client ID', right_on='Client ID')
    JoinTest2.head()

Out[365]:

Client ID

Age

Gender

County

Income Group

SNo

Population

Density (/ km2)

Rank

Province

...

Held Loan previously

ProductsAmount

Client\_x

Num Transactions

Last TXN Amount

Merchant Code

Last Transaction Narrative

Client\_y

TxnAmount

Loan Flag

0

10001.0

59.0

1.0

Cork

10001 - 40000

4

519032.0

69.0

4

Munster

...

0.0

4.0

10001.0

2.0

12.59

7375.0

MYWHEELS IE DUBLIN 2

10001.0

22.0

NaN

1

10007.0

43.0

0.0

Cork

10001 - 40000

4

519032.0

69.0

4

Munster

...

0.0

1.0

10007.0

0.0

674.60

3742.0

CLUB MEDITERRANEE L.AIDIPSOU EV

10007.0

31.0

NaN

2

10010.0

52.0

0.0

Cork

10001 - 40000

4

519032.0

69.0

4

Munster

...

0.0

2.0

10010.0

36.0

312.98

5462.0

O'HEHIRS ATHLONE

10010.0

32.0

NaN

3

10014.0

52.0

1.0

Cork

10001 - 40000

4

519032.0

69.0

4

Munster

...

0.0

2.0

10014.0

0.0

1102.64

5261.0

GARDENING IMPULSE LIMITEDDRINAGH

10014.0

6.0

NaN

4

10032.0

55.0

1.0

Cork

10001 - 40000

4

519032.0

69.0

4

Munster

...

1.0

1.0

10032.0

12.0

1148.15

7342.0

RENTOKIL PEST CONTROL NAAS

10032.0

42.0

NaN

5 rows × 21 columns

In [366]:

    JoinTesting = JoinTest2.copy()
    JoinTesting.head()

Out[366]:

Client ID

Age

Gender

County

Income Group

SNo

Population

Density (/ km2)

Rank

Province

...

Held Loan previously

ProductsAmount

Client\_x

Num Transactions

Last TXN Amount

Merchant Code

Last Transaction Narrative

Client\_y

TxnAmount

Loan Flag

0

10001.0

59.0

1.0

Cork

10001 - 40000

4

519032.0

69.0

4

Munster

...

0.0

4.0

10001.0

2.0

12.59

7375.0

MYWHEELS IE DUBLIN 2

10001.0

22.0

NaN

1

10007.0

43.0

0.0

Cork

10001 - 40000

4

519032.0

69.0

4

Munster

...

0.0

1.0

10007.0

0.0

674.60

3742.0

CLUB MEDITERRANEE L.AIDIPSOU EV

10007.0

31.0

NaN

2

10010.0

52.0

0.0

Cork

10001 - 40000

4

519032.0

69.0

4

Munster

...

0.0

2.0

10010.0

36.0

312.98

5462.0

O'HEHIRS ATHLONE

10010.0

32.0

NaN

3

10014.0

52.0

1.0

Cork

10001 - 40000

4

519032.0

69.0

4

Munster

...

0.0

2.0

10014.0

0.0

1102.64

5261.0

GARDENING IMPULSE LIMITEDDRINAGH

10014.0

6.0

NaN

4

10032.0

55.0

1.0

Cork

10001 - 40000

4

519032.0

69.0

4

Munster

...

1.0

1.0

10032.0

12.0

1148.15

7342.0

RENTOKIL PEST CONTROL NAAS

10032.0

42.0

NaN

5 rows × 21 columns

In [367]:

    JoinTesting = JoinTesting[JoinTesting.columns.difference(["SNo","Client_x","Client_y","Last Transaction Narrative","Province","Merchant Code"])]

In [368]:

    JoinTesting[pd.isnull(JoinTesting['Client ID'])]

Out[368]:

Age

Change

Client ID

County

Density (/ km2)

Gender

Held Loan previously

Income Group

Last TXN Amount

Loan Flag

Num Transactions

Population

ProductsAmount

Rank

TxnAmount

2000

NaN

0.0180

NaN

Antrim

202.9

NaN

NaN

NaN

NaN

NaN

NaN

618108.0

NaN

2

NaN

2001

NaN

0.0872

NaN

Down

215.6

NaN

NaN

NaN

NaN

NaN

NaN

531665.0

NaN

3

NaN

2002

NaN

0.0478

NaN

Londonderry

119.1

NaN

NaN

NaN

NaN

NaN

NaN

247132.0

NaN

8

NaN

2003

NaN

0.0837

NaN

Tyrone

54.5

NaN

NaN

NaN

NaN

NaN

NaN

177986.0

NaN

13

NaN

2004

NaN

0.0726

NaN

Armagh

131.8

NaN

NaN

NaN

NaN

NaN

NaN

174792.0

NaN

14

NaN

2005

NaN

0.1030

NaN

Wexford

61.2

NaN

NaN

NaN

NaN

NaN

NaN

145320.0

NaN

18

NaN

2006

NaN

0.0859

NaN

Westmeath

46.7

NaN

NaN

NaN

NaN

NaN

NaN

86164.0

NaN

26

NaN

2007

NaN

0.0739

NaN

Sligo

35.5

NaN

NaN

NaN

NaN

NaN

NaN

65393.0

NaN

31

NaN

2008

NaN

0.0901

NaN

Roscommon

25.0

NaN

NaN

NaN

NaN

NaN

NaN

64065.0

NaN

32

NaN

2009

NaN

0.0633

NaN

Fermanagh

36.1

NaN

NaN

NaN

NaN

NaN

NaN

61170.0

NaN

33

NaN

2010

NaN

0.0630

NaN

Dun Laoghaire-Rathdown

1620.1

NaN

NaN

NaN

NaN

NaN

NaN

206261.0

NaN

10

NaN

2011

NaN

0.0626

NaN

South Tipperary

39.2

NaN

NaN

NaN

NaN

NaN

NaN

88432.0

NaN

25

NaN

2012

NaN

0.1417

NaN

Fingal

600.6

NaN

NaN

NaN

NaN

NaN

NaN

273991.0

NaN

5

NaN

2013

NaN

0.0661

NaN

North Tipperary

34.4

NaN

NaN

NaN

NaN

NaN

NaN

70322.0

NaN

30

NaN

2014

NaN

0.0740

NaN

South Dublin

1190.6

NaN

NaN

NaN

NaN

NaN

NaN

265205.0

NaN

6

NaN

In [369]:

    JoinTesting = JoinTesting[np.logical_not(pd.isnull(JoinTesting['Client ID']))]

In [370]:

    JoinTesting['Income Group'].unique()

Out[370]:

    array([u'10001 - 40000', u'60001 - 100000', u'40001 - 60000', u'0 - 10000',
           u'100000+'], dtype=object)

In [371]:

    JoinTesting.ix[JoinTesting['Income Group'] == "0 - 10000", "Income Group"] = 1
    JoinTesting.ix[JoinTesting['Income Group'] == "10001 - 40000", "Income Group"] = 2
    JoinTesting.ix[JoinTesting['Income Group'] == "40001 - 60000", "Income Group"] = 3
    JoinTesting.ix[JoinTesting['Income Group'] == "60001 - 100000", "Income Group"] = 4
    JoinTesting.ix[JoinTesting['Income Group'] == "100000+", "Income Group"] = 5
    JoinTesting['Income Group'] = map (lambda x: int(x), JoinTesting['Income Group'])

In [372]:

    JoinTesting = JoinTesting[JoinTesting.columns.difference(["Rank","Population","Change"])]

    X = JoinTesting[JoinTesting.columns.difference(['County','Client ID','Loan Flag'])]

In [373]:

    predictions = model_Final.predict(JoinTesting[JoinTesting.columns.difference(['County','Client ID','Loan Flag'])])
    prob = model_Final.predict_proba(JoinTesting[JoinTesting.columns.difference(['County','Client ID','Loan Flag'])])

In [374]:

    JoinTesting["loan_predicted"] = predictions
    JoinTesting["prob"] = prob[:,1]

In [375]:

    JoinTesting["Loan Likelihood"] = map (lambda x: classLike(x), JoinTesting["prob"])

In [376]:

    JoinTesting.sort_values("Client ID").head()

Out[376]:

Age

Client ID

County

Density (/ km2)

Gender

Held Loan previously

Income Group

Last TXN Amount

Loan Flag

Num Transactions

ProductsAmount

TxnAmount

loan\_predicted

prob

Loan Likelihood

0

59.0

10001.0

Cork

69.0

1.0

0.0

2

12.59

NaN

2.0

4.0

22.0

0.0

0.006667

1

260

27.0

10002.0

Kerry

30.1

1.0

0.0

2

30.00

NaN

0.0

4.0

11.0

0.0

0.006667

1

296

58.0

10003.0

Louth

148.7

0.0

0.0

2

1003.01

NaN

28.0

2.0

9.0

0.0

0.380000

1

350

45.0

10004.0

Dublin

1380.8

1.0

0.0

4

873.25

NaN

31.0

2.0

34.0

0.0

0.100000

1

351

21.0

10005.0

Dublin

1380.8

0.0

0.0

3

926.75

NaN

12.0

1.0

38.0

0.0

0.300000

1

In [377]:

    JoinTesting = JoinTesting.sort_values("Client ID")

    JoinTesting["Client ID"] = map(lambda x: int(x), JoinTesting["Client ID"])
    JoinTesting["loan_predicted"] = map(lambda x: int(x), JoinTesting["loan_predicted"])
    JoinTesting[""] = ""

In [378]:

    JoinTesting["Loan Flag"] = JoinTesting["loan_predicted"] 

In [379]:

    Output = JoinTesting[["Client ID","Loan Flag",""]]

In [380]:

    Output.to_csv("Test - Purchased Loan Flag.csv", index=False)

In [381]:

    Output["Loan Flag"].sum()

Out[381]:

    354

In [382]:

    OutputL = JoinTesting[["Client ID","Loan Likelihood",""]]
    OutputL.to_csv("Test - Loan Likelihood.csv", index=False)

In [ ]:

     
