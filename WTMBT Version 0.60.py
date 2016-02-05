"""
The World Trade Model in Python using Pulp
Date: 2-5-2016

Version: 0.60
Benefit of Trade: No
No Trade Sectors: No
Bilateral Trade: Yes
RCOT: Yes
AMC: No

"""

# Imports packages not built into Python
import pulp
#pulp.pulpTestAll()
import numpy as np
import pandas as pd


Indices = "WTMBT_Indices.xlsx"
ParametersEmpty = 'WTMBT_Parameters_Empty.xlsx'
Parameters = "WTMBT_Parameters.xlsx"
NTMParameters = "NTM_Parameters.xlsx"
Results = 'WTMBT_Results.xlsx'
test = "test.csv"

#Define sets and create dataframes for Excel

#Sets
data = pd.ExcelFile(Indices)
sets = data.parse(sheetname="Des", header=8, parse_cols="B,D,F,G,I,J,L")
Regions = sets.ix[:,0].dropna().values
Sectors = sets.ix[:,1].dropna().values
Technologies = sets.ix[:,2].dropna().values
Transport_Sectors = sets.ix[:,3].dropna().values
No_Trade_Sectors = sets.ix[:,4].dropna().values
Factors_k = sets.ix[:,5].dropna().values
Factors_l = sets.ix[:,6].dropna().values

imax = Regions.size
jmax = imax
nmax = Sectors.size
mmax = nmax
tmax = Technologies.size
ntmax = No_Trade_Sectors.size
kmax = Factors_k.size
lmax = Factors_l.size

xmax = tmax*imax
exmax = nmax*imax*jmax
#immax = nmax*imax*jmax
pmax = nmax*imax
pntmax = nmax*imax
rmax = kmax*imax

# Define parameters and indices
Aindex = pd.MultiIndex.from_product([Regions,Sectors,Technologies], names=['Regions','Sectors','Technologies'])
Intermediate_Inputs = pd.Series(data='', index=Aindex).unstack()
Fkindex = pd.MultiIndex.from_product([Regions,Factors_k,Technologies], names=['Regions','Factor Inputs','Technologies'])
Factor_Inputs = pd.Series(data='', index=Fkindex).unstack()
Flindex = pd.MultiIndex.from_product([Regions,Factors_l,Technologies], names=['Regions','Factor Input Costs','Technologies'])
Factor_Input_Costs = pd.Series(data='', index=Flindex).unstack()
yindex = pd.MultiIndex.from_product([Regions,Sectors], names=['Regions','Sectors'])
Final_Demand = pd.Series(data='', index=yindex).unstack()
findex = pd.MultiIndex.from_product([Regions,Factors_k], names=['Regions','Factor Inputs'])
Endowments = pd.Series(data='', index=findex).unstack()
piindex = pd.MultiIndex.from_product([Regions,Factors_l], names=['Regions','Factor Input Costs'])
Factor_Prices = pd.Series(data='', index=piindex).unstack()
Iindex = pd.MultiIndex.from_product([Sectors,Sectors], names=['Sectors','Sectors'])
Identity = pd.Series(data='', index=Iindex).unstack()
Itindex = pd.MultiIndex.from_product([Sectors,Technologies], names=['Sectors','Technologies'])
Identity_RCOT = pd.Series(data='', index=Itindex).unstack()
Transport_Weights = pd.Series(data='', index=Iindex).unstack()
Tindex = pd.MultiIndex.from_product([Regions,Regions], names=['Regions', 'Regions'])
Transport_Distances = pd.Series(data='', index=Tindex).unstack() 

xindex = pd.MultiIndex.from_product([Technologies,Regions], names=['Technologies','Regions'])
exindex = pd.MultiIndex.from_product([Sectors,Regions,Regions], names=['Sectors','Regions','Regions'])
#imindex = pd.MultiIndex.from_product([Sectors,Regions,Regions], names=['Sectors','Regions','Regions'])
pindex = pd.MultiIndex.from_product([Sectors,Regions], names=['Sectors','Regions'])
pntindex = pd.MultiIndex.from_product([Sectors,Regions], names=['Sectors','Regions'])
rindex = pd.MultiIndex.from_product([Factors_k,Regions], names=['Factor Inputs','Regions'])
alphaindex = pd.MultiIndex.from_product([Regions], names=['Regions'])
ntrindex = pd.MultiIndex.from_product([No_Trade_Sectors,Regions], names=['No Trade Sectors','Regions'])

#Define indices
idx = np.arange(imax)
jdx = np.arange(jmax)
ndx = np.arange(nmax)
mdx = np.arange(mmax)
tdx = np.arange(tmax)
ntdx = np.arange(ntmax)
kdx = np.arange(kmax)
ldx = np.arange(lmax)

xdx = np.arange(xmax)
exdx = np.arange(exmax)
#imdx = np.arange(immax)
pdx = np.arange(pmax)
pntdx = np.arange(pntmax)
rdx = np.arange(rmax)

#Export Dataframes to excel
writer = pd.ExcelWriter(ParametersEmpty, engine='xlsxwriter')
with writer as writer:
    Intermediate_Inputs.to_excel(writer, sheet_name='A', merge_cells=False)
    Final_Demand.to_excel(writer, sheet_name='y')
    Factor_Inputs.to_excel(writer, sheet_name='Fk', merge_cells=False)
    Factor_Input_Costs.to_excel(writer, sheet_name='Fl', merge_cells=False)
    Endowments.to_excel(writer, sheet_name='f')
    Factor_Prices.to_excel(writer, sheet_name='pi')
    Identity.to_excel(writer, sheet_name='I')
    Identity_RCOT.to_excel(writer, sheet_name='It')
    Transport_Weights.to_excel(writer, sheet_name='W')
    Transport_Distances.to_excel(writer, sheet_name='D')
writer.save()

#Import full dataframes back to python
data = pd.ExcelFile(Parameters)
Intermediate_Inputs = data.parse(sheetname="A", header=0, parse_cols=(2+tmax), index_col=[0,1]).stack().reindex(index=Aindex).unstack()
Factor_Inputs = data.parse(sheetname="Fk", header=0, parse_cols=(2+tmax), index_col=[0,1]).stack().reindex(index=Fkindex).unstack()
Factor_Input_Costs = data.parse(sheetname="Fl", header=0, parse_cols=(2+tmax), index_col=[0,1]).stack().reindex(index=Flindex).unstack()
Final_Demand = data.parse(sheetname="y", header=0, parse_cols=(1+nmax)).stack().reindex(index=yindex).unstack()
Endowments = data.parse(sheetname="f", header=0, parse_cols=(1+kmax)).stack().reindex(index=findex).unstack()
Factor_Prices = data.parse(sheetname="pi", header=0, parse_cols=(1+lmax)).stack().reindex(index=piindex).unstack()
Identity = data.parse(sheetname="I", header=0, parse=(1+nmax)).stack().reindex(index=Iindex).unstack()
Identity_RCOT = data.parse(sheetname="It", header=0, parse=(1+tmax)).stack().reindex(index=Itindex).unstack()
Transport_Weights = data.parse(sheetname="W", header=0, parse=(1+nmax)).stack().reindex(index=Iindex).unstack()
Transport_Distances = data.parse(sheetname="D", header=0, parse=(1+imax)).stack().reindex(index=Tindex).unstack()

#Build no-trade structures
#Intermediate_Inputs_NT = Intermediate_Inputs.loc(axis=0)[:,No_Trade_Sectors]
#Identity_RCOT_NT = Identity_RCOT.loc(axis=0)[No_Trade_Sectors]
#Final_Demand_NT = Final_Demand.transpose()
#Final_Demand_NT = Final_Demand_NT.loc(axis=0)[No_Trade_Sectors]
#writer = pd.ExcelWriter(NTMParameters, engine='xlsxwriter')
#with writer as writer:
#    Intermediate_Inputs_NT.to_excel(writer, sheet_name='Ant', merge_cells=False)
#    Identity_RCOT_NT.to_excel(writer, sheet_name='Itnt', merge_cells=False)
#    Final_Demand_NT.to_excel(writer, sheet_name='ynt', merge_cells=False)
#writer.save()
#data = pd.ExcelFile(NTMParameters)
#Intermediate_Inputs_NT = data.parse(sheetname="Ant", header=0, parse_cols=(2+tmax), index_col=[0,1])
#Identity_RCOT_NT = data.parse(sheetname="Itnt", header=0, parse_cols=(1+tmax), index_col=[0])
#Final_Demand_NT = data.parse(sheetname="ynt", header=0, parse_cols=(1+tmax), index_col=[0])

#Change data to numpy arrays for PuLP
A = Intermediate_Inputs.to_panel().values.transpose(1,2,0)
Fk = Factor_Inputs.to_panel().values
Fl = Factor_Input_Costs.to_panel().values
y = Final_Demand.values
f = Endowments.values
pi = Factor_Prices.values
I = Identity.values
It = Identity_RCOT.values
#Ant = Intermediate_Inputs_NT.to_panel().values.transpose(1,2,0)
#Itnt = Identity_RCOT_NT.values
#ynt = Final_Demand_NT.values
W = Transport_Weights.values
D = Transport_Distances.values

#Build (It-A) matrix
IA = It - A
#IAnt = Itnt - Ant

#Build Transport Matrix
T = np.ones((imax,jmax,nmax,nmax))
for i in idx:
    for j in jdx:
        for n in ndx:
            for m in ndx:
                T[i,j,n,m] = np.sum([D[i,j] * W[n,m]])
        
#Build (I-T) matrix
IT = I - T 

#Check numpy arrays for NaN
#np.isnan(np.sum(A))

#PULP No Trade Model PRIMAL (Quantity model)

#LP Define
NTM_Quantity = pulp.LpProblem("NTM_Quanity", pulp.LpMinimize)

#Variables
x_nt = pulp.LpVariable.dicts("NTM_Production", xindex, 0)

#Objective Function
NTM_Quantity += np.sum([pi[i,l] * Fl[t,i,l] * x_nt[Technologies[t], Regions[i]] for t in tdx for i in idx for l in ldx])

#Constraints
for n in ndx:
    for i in idx:
        NTM_Quantity += np.sum([IA[i, n, t] * x_nt[Technologies[t],Regions[i]] for t in tdx]) >= y[i, n]
    
#Solver
#NTM_Quantity.solve()
NTM_Quantity.solve(pulp.CPLEX_PY())

#Results  
Production_NT = pd.Series('', index=xindex)
for v in xindex:
  Production_NT[v] = x_nt[v].varValue 
Production_NT = Production_NT.unstack()
x_nt = Production_NT.values
Production_NT = Production_NT.assign(Total = Production_NT.sum(axis=1))


#PULP No Trade Model DUAL (Price model)

#LP Define
NTM_Price = pulp.LpProblem("NTM_Price", pulp.LpMaximize)

#Variables
p_nt = pulp.LpVariable.dicts("NTM_Prices", pntindex, 0)

#Objective Function
NTM_Price += np.sum([p_nt[Sectors[n],Regions[i]] * y[i,n] for n in ndx for i in idx])

#Constraints
for t in tdx:
    for i in idx:
        NTM_Price += np.sum([IA[i, n, t] * p_nt[Sectors[n],Regions[i]] for n in ndx]) <= np.sum([Fl[t,i,l] * pi[i,l] for l in ldx])
    
#Solver
#NTM_Price.solve()
NTM_Price.solve(pulp.CPLEX_PY())

#Results
Prices_NT = pd.Series('', index=pntindex)
for v in pntindex:
  Prices_NT[v] = p_nt[v].varValue 
Prices_NT = Prices_NT.unstack()
p_nt = Prices_NT.values




#PULP World Trade Model with Bilateral Trade PRIMAL (Quantity Model)

#LP Define
WTMBT_Quantity = pulp.LpProblem("WTMBT_Quantity", pulp.LpMinimize)

#Variables
x = pulp.LpVariable.dicts("Production", xindex, lowBound=0)
ex = pulp.LpVariable.dicts("Exports", exindex, lowBound=0)
#im = pulp.LpVariable.dicts("Imports", imindex, 0)

#Objective Function
#WTMBT_Quantity += np.sum([pi[i,l] * Fl[t,i,l] * x[Technologies[t], Regions[i]] for t in tdx for i in idx for l in ldx])
WTMBT_Quantity += np.sum([pi[i,l] * Fl[t,i,l] * x[Technologies[t], Regions[i]] for t in tdx for i in idx for l in ldx])

#Constraints 
for n in ndx:
    for i in idx:
        WTMBT_Quantity += np.sum([IA[i, n, t] * x[Technologies[t],Regions[i]] for t in tdx]) + np.sum([IT[i,j,n,m] * ex[Sectors[m], Regions[j], Regions[i]] for m in ndx for j in jdx]) - np.sum([I[n,m] * ex[Sectors[n], Regions[i], Regions[j]] for m in ndx for j in jdx]) >= y[i,n]

for k in kdx:
    for i in idx:
        WTMBT_Quantity += np.sum([-Fk[t, i, k] * x[Technologies[t],Regions[i]] for t in tdx]) >= -f[i, k]

#Solver
#WTMBT_Quantity.solve()
WTMBT_Quantity.solve(pulp.CPLEX_PY())

#Results
Production = pd.Series('', index=xindex)
for v in xindex:
  Production[v] = x[v].varValue 
Production = Production.unstack()
#ProductionOnes = Production
#ProductionOnes[ProductionOnes > 0] = 1

x = Production.values
Sum = Production.sum(axis=1)
Production = Production.assign(Total = Production.sum(axis=1))


Exports = pd.Series('', index=exindex)
for v in exindex:
    Exports[v] = ex[v].varValue
Exports = Exports.unstack()
ex = Exports.to_panel().values

Exports_Sum = Exports.sum(axis=1).unstack()
Imports_Sum = Exports.sum(axis=0, level=0)
ImportsPDEC = Imports_Sum.stack().swaplevel(0,1)
Net_Exports = Exports_Sum - Imports_Sum
Exports_Sum["Exports"] = "Exports"
Imports_Sum["Imports"] = "Imports"
Net_Exports["Net Exports"] = "Net Exports"
Exports_Sum = Exports_Sum.set_index("Exports", append=True)
Imports_Sum = Imports_Sum.set_index("Imports", append=True)
Net_Exports = Net_Exports.set_index("Net Exports", append=True)
Exports_Sum.index = Exports_Sum.index.swaplevel(0,1)
Imports_Sum.index = Imports_Sum.index.swaplevel(0,1)
Net_Exports.index = Net_Exports.index.swaplevel(0,1)

Exports = pd.concat([Exports, Exports_Sum])
Exports = pd.concat([Exports, Imports_Sum])
Exports = pd.concat([Exports, Net_Exports])
Exports = Exports.assign(Total = Exports.sum(axis=1))


#PULP World Trade Model with Bilateral Trade DUAL (Price Model)

#LP Define
WTMBT_Price = pulp.LpProblem("WTMBT_Price", pulp.LpMaximize)

#Variables
p = pulp.LpVariable.dicts("Prices", pindex, lowBound=0)
r = pulp.LpVariable.dicts("Rents", rindex, lowBound=0)

#Objective Function
WTMBT_Price += np.sum([p[Sectors[n],Regions[i]] * y[i,n] for i in idx for n in ndx]) - np.sum([r[Factors_k[k], Regions[i]] * f[i,k] for k in kdx for i in idx])

#Constraints
for t in tdx:
    for i in idx:
        WTMBT_Price += np.sum([IA[i, n, t] * p[Sectors[n],Regions[i]] for n in ndx]) - np.sum([Fk[t,i,k] * r[Factors_k[k], Regions[i]] for k in kdx]) <= np.sum([Fl[t,i,l] * pi[i,l] for l in ldx])
         
for m in ndx:
    for i in idx:
        for j in jdx:
            WTMBT_Price += np.sum([I[n,m] * p[Sectors[n],Regions[i]] for n in ndx]) - np.sum([T[i,j,n,m] * p[Sectors[n],Regions[i]] for n in ndx]) - np.sum([I[n,m] * p[Sectors[n],Regions[j]] for n in ndx]) <= 0  

#Solver
#WTMBT_Price.solve()
WTMBT_Price.solve(pulp.CPLEX_PY())

#Results
Prices = pd.Series('', index=pindex)
for v in pindex:
  Prices[v] = p[v].varValue
p = Prices.values
Prices = Prices.unstack()
p = Prices.values
pt = np.dot(It.T, p) 
Prices_t = pd.DataFrame(data=pt, index=Technologies, columns=Regions)

Factor_Rents = pd.Series('', index=rindex)
for v in rindex:
  Factor_Rents[v] = r[v].varValue 
Factor_Rents = Factor_Rents.unstack().replace(to_replace='None',value=0).replace(to_replace='NaN',value=0)
r = Factor_Rents.values 

#Print status of all model runs
print "Status:", pulp.LpStatus[NTM_Quantity.status]
print pulp.value(NTM_Quantity.objective)       
print "Status:", pulp.LpStatus[NTM_Price.status]
print pulp.value(NTM_Price.objective)  
print "Status:", pulp.LpStatus[WTMBT_Quantity.status]  
print pulp.value(WTMBT_Quantity.objective) 
print "Status:", pulp.LpStatus[WTMBT_Price.status]
print pulp.value(WTMBT_Price.objective)

#Calculate additional results

#Factor Use
Fkx = np.ones((tmax,imax,kmax))
for t in tdx:
    for i in idx:
        for k in kdx:
            Fkx[t,i,k] = Fk[t,i,k] * x[t,i]
Factor_Use = pd.Panel(data=Fkx, items=Technologies, major_axis=Regions, minor_axis=Factors_k)
Factor_Use = Factor_Use.to_frame()
#Factor_Use = Factor_Use.to_frame().stack().reset_index()
#Factor_Use.columns = ['Regions','Factors','Technologies','']
#Factor_Use = Factor_Use.set_index(['Regions','Factors','Technologies']).unstack()

Region_Sum = Factor_Use.sum(axis=0, level=1)
Region_Sum["Regions"] = "World"
Region_Sum = Region_Sum.set_index("Regions", append=True)
Region_Sum.index = Region_Sum.index.swaplevel(0,1)

Factor_Use = pd.concat([Factor_Use, Region_Sum])
Factor_Use = Factor_Use.assign(Total = Factor_Use.sum(axis=1)).sortlevel(level=1, axis=0)
            
#Factor_Costs
Flpi = np.ones((lmax,imax,tmax))
for t in tdx:
    for i in idx:
        for l in ldx:
            Flpi[l,i,t] = Fl[t,i,l]*pi[i,l]                 
Factor_Costs = pd.Panel(data=Flpi, items=Factors_l, major_axis=Regions, minor_axis=Technologies)        
Factor_Costs = Factor_Costs.to_frame().sortlevel(level=1, axis=0)

#Intermediate_Production_Costs
Ap = np.ones((nmax,imax,tmax))
Ap2 = np.ones((imax,tmax))
for t in tdx:
    for i in idx:
        for n in ndx:
            Ap[n,i,t]  = A[i,n,t]*p[n,i]
Intermediate_Costs = pd.Panel(data=Ap, items=Sectors, major_axis=Regions, minor_axis=Technologies)        
Intermediate_Costs = Intermediate_Costs.to_frame().sortlevel(level=1, axis=0)


#Transport_Costs
Tp = np.ones((imax,jmax,mmax))
for i in idx:
    for j in jdx:
        for n in ndx:
            for m in ndx:
                Tp[i,j,m] = np.sum([T[i,j,n,m] * p[n,i] for n in ndx])
Transport_Costs = pd.Panel(data=Tp, items=Regions, major_axis=Regions, minor_axis=Sectors)
#Transport_Costs = Transport_Costs.transpose(0,2,1)
Transport_Costs = Transport_Costs.to_frame().sortlevel(level=1, axis=0)

#exones = np.ones((jmax,tmax,imax))
#ones = np.ones((jmax,tmax,imax))
#exones = ex * ones
#for i in idx:
#    for j in jdx:
#        for n in ndx:
#            exones[j,t,i] = ex[j,t,i] * ones[j,t,i]

#ImTp = np.ones((imax,jmax,mmax))
#for i in idx:
#    for j in jdx:
#        for n in ndx:
#            for m in ndx:
#                ImTp[i,j,m] = np.sum([ex[j,m,i] * T[i,j,n,m] * p[n,i] for n in ndx])
#Total_Transport_Costs = pd.Panel(data=ImTp, items=Regions, major_axis=Regions, minor_axis=Sectors)
#Total_Transport_Costs = Total_Transport_Costs.to_frame().sortlevel(level=1, axis=0)

#pTp = np.ones((imax,jmax,mmax))
#for i in idx:
#    for j in jdx:
#            for m in ndx:
#                pTp[i,j,m] = np.sum([It[n,m] * p[n,j] for n in ndx]) + np.sum([T[i,j,n,m] * p[n,i] for n in ndx])
#Import_Prices = pd.Panel(data=pTp, items=Regions, major_axis=Regions, minor_axis=Sectors)
#Import_Prices = Import_Prices.to_frame().sortlevel(level=1, axis=0)


#Factor_Rent_Payments
Fkr = np.ones((kmax,imax,tmax))
for t in tdx:
    for i in idx:
        for k in kdx:
            Fkr[k,i,t]  = Fk[t,i,k]*r[k,i]
Factor_Rent_Payments = pd.Panel(data=Fkr, items=Factors_k, major_axis=Regions, minor_axis=Technologies)        
Factor_Rent_Payments = Factor_Rent_Payments.to_frame().sortlevel(level=1, axis=0)

#xFkr = np.ones((kmax,imax,tmax))
#for t in tdx:
#    for i in idx:
#        for k in kdx:
#            xFkr[k,i,t]  = r[k,i] * Fk[t,i,k] * x[t,i]
#Total_Factor_Rent_Payments = pd.Panel(data=xFkr, items=Factors_k, major_axis=Regions, minor_axis=Technologies).to_frame().sortlevel(level=1, axis=0)
#rFkx = xFkr.transpose(2,1,0)
#Factor_Rent_Payments_Total = pd.Panel(data=rFkx, items=Technologies, major_axis=Regions, minor_axis=Factors_k).to_frame().sortlevel(level=1, axis=0)


#Pdec
Pdec = pd.concat([Factor_Costs, Intermediate_Costs, Factor_Rent_Payments], axis=1)
Totals = Production.stack().swaplevel(0,1).sortlevel('Regions')
Totals.name = 'Production'
Totals = Totals.to_frame()
Totals = Totals.assign(Imports = ImportsPDEC)
Totals = Totals.assign(Prices = Prices_t.unstack())
Totals = Totals.assign(Pdec_Total = Pdec.sum(axis=1))
Totals = Totals.assign(Factor_Costs_Total = Factor_Costs.sum(axis=1))
Totals = Totals.assign(Intermediate_Costs_Total = Intermediate_Costs.sum(axis=1))
Totals = Totals.assign(Factor_Rent_Payments_Total = Factor_Rent_Payments.sum(axis=1))
Totals = Totals.assign(Transport_Costs = Transport_Costs.sum(axis=1))
Totals = Totals.sortlevel(level=1, axis=0)
#Pdec = pd.concat([Totals,Pdec],axis=1)  




#Export results
writer = pd.ExcelWriter(Results, engine='xlsxwriter')
with writer as writer:
    
    Production_NT.to_excel(writer, sheet_name='x_nt', merge_cells=False)
    Prices_NT.to_excel(writer, sheet_name='p_nt', merge_cells=False)
    Production.to_excel(writer, sheet_name='x', merge_cells=False)
    Exports.to_excel(writer, sheet_name='ex', merge_cells=False)
    Prices.to_excel(writer, sheet_name='p')
    Factor_Rents.to_excel(writer, sheet_name='r')
    
    Factor_Use.to_excel(writer, sheet_name='Fkx', merge_cells=False)
    #Factor_Rent_Payments_Total.to_excel(writer, sheet_name='rFkx', merge_cells=False)
    
    Factor_Costs.to_excel(writer, sheet_name='Flpi',merge_cells=False)
    Intermediate_Costs.to_excel(writer, sheet_name='Ap', merge_cells=False)
    Factor_Rent_Payments.to_excel(writer, sheet_name='Fkr', merge_cells=False)
    #Total_Factor_Rent_Payments.to_excel(writer, sheet_name='xFkr', merge_cells=False)
    Totals.to_excel(writer, sheet_name='Pdec', merge_cells=False)
    Transport_Costs.to_excel(writer, sheet_name='Tp', merge_cells=False)
    #Total_Transport_Costs.to_excel(writer, sheet_name='imTp', merge_cells=False)
    #Import_Prices.to_excel(writer, sheet_name='pTp', merge_cells=False)
    
writer.save()

Productionstack = Production.stack()
Productionstack.to_csv(test,sep=',')  
