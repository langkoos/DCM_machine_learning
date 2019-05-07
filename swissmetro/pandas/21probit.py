import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio

pandas = pd.read_table("swissmetro.dat")
database = db.Database("swissmetro",pandas)

# The Pandas data structure is available as database.data. Use all the
# Pandas functions to invesigate the database
#print(database.data.describe())

from headers import *

# Removing some observations can be done directly using pandas.
#remove = (((database.data.PURPOSE != 1) & (database.data.PURPOSE != 3)) | (database.data.CHOICE == 0))
#database.data.drop(database.data[remove].index,inplace=True)

# Here we use the "biogeme" way for backward compatibility As we
# estimate a binary model, we remove observations where Swissmetro was
# chosen (CHOICE == 2). We also remove observations where one of the
# two alternatives is not available.

CAR_AV_SP =  DefineVariable('CAR_AV_SP',CAR_AV  * (  SP   !=  0  ),database)
TRAIN_AV_SP =  DefineVariable('TRAIN_AV_SP',TRAIN_AV  * (  SP   !=  0  ),database)
exclude = (TRAIN_AV_SP == 0) + (CAR_AV_SP == 0) + ( CHOICE == 2 ) + (( PURPOSE != 1 ) * (  PURPOSE   !=  3  ) + ( CHOICE == 0 )) > 0


database.remove(exclude)



ASC_CAR = Beta('ASC_CAR',1,None,None,0)
ASC_TRAIN = Beta('ASC_TRAIN',1,None,None,0)
ASC_SM = Beta('ASC_SM',1,None,None,1)
B_TIME = Beta('B_TIME',1,None,None,0)
B_COST = Beta('B_COST',1,None,None,0)



SM_COST =  SM_CO   * (  GA   ==  0  ) 
TRAIN_COST =  TRAIN_CO   * (  GA   ==  0  )

TRAIN_TT_SCALED = DefineVariable('TRAIN_TT_SCALED',\
                                 TRAIN_TT / 100.0,database)
TRAIN_COST_SCALED = DefineVariable('TRAIN_COST_SCALED',\
                                   TRAIN_COST / 100,database)
SM_TT_SCALED = DefineVariable('SM_TT_SCALED', SM_TT / 100.0,database)
SM_COST_SCALED = DefineVariable('SM_COST_SCALED', SM_COST / 100,database)
CAR_TT_SCALED = DefineVariable('CAR_TT_SCALED', CAR_TT / 100,database)
CAR_CO_SCALED = DefineVariable('CAR_CO_SCALED', CAR_CO / 100,database)

# We estimate a binary probit model. There are only two alternatives.
V1 = B_TIME * TRAIN_TT_SCALED + \
     B_COST * TRAIN_COST_SCALED
V3 = ASC_CAR + \
     B_TIME * CAR_TT_SCALED + \
     B_COST * CAR_CO_SCALED

# Associate choice probability with the numbering of alternatives

P = {1: bioNormalCdf(V1-V3),
     3: bioNormalCdf(V3-V1)}



prob = Elem(P,CHOICE)

biogeme  = bio.BIOGEME(database,log(prob),numberOfThreads=1)
biogeme.modelName = "21probit"
#results = biogeme.checkDerivatives(logg=True)
results = biogeme.estimate()


print("Results=",results)

#[ 0.0027692   0.00638408 -0.00252952] [[ 0.01438453  0.03438956 -0.01500251]
# [ 0.05682623  0.13743163 -0.06133221]
# [ 0.00272752  0.00829416 -0.00771619]]
