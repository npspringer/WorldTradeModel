"""
The World Trade Model with Rectangular Choice-in-Technology in Python using Pulp
Date: 2-5-2016

Version: 0.60
Benefit of Trade: No
No Trade Sectors: No
Bilateral Trade: No
RCOT: Yes
AMC: No

"""

# Imports packages not built into Python
import pulp
#pulp.pulpTestAll()
import numpy as np
import pandas as pd




#Define sets and create dataframes for Excel

#Sets
data = pd.ExcelFile("WTM_Indices.xlsx")
sets = data.parse(sheetname="Des", header=8, parse_cols="B,D,F,G,I,J,L")
Regions = sets.ix[:,0].dropna().values
Sectors = sets.ix[:,1].dropna().values
Technologies = sets.ix[:,2].dropna().values
Transport_Sectors = sets.ix[:,3].dropna().values
No_Trade_Sectors = sets.ix[:,4].dropna().values
Factors_k = sets.ix[:,5].dropna().values
Factors_l = sets.ix[:,6].dropna().values

imax = Regions.size
nmax = Sectors.size
tmax = Technologies.size
ntmax = No_Trade_Sectors.size
kmax = Factors_k.size
lmax = Factors_l.size

xmax = tmax*imax
pmax = nmax
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

xindex = pd.MultiIndex.from_product([Technologies,Regions], names=['Technologies','Regions'])
pindex = pd.MultiIndex.from_product([Sectors], names=['Sectors'])
pntindex = pd.MultiIndex.from_product([Sectors,Regions], names=['Sectors','Regions'])
rindex = pd.MultiIndex.from_product([Factors_k,Regions], names=['Factor Inputs','Regions'])
alphaindex = pd.MultiIndex.from_product([Regions], names=['Regions'])
ntrindex = pd.MultiIndex.from_product([No_Trade_Sectors,Regions], names=['No Trade Sectors','Regions'])

#Define indices
idx = np.arange(imax)
ndx = np.arange(nmax)
tdx = np.arange(tmax)
ntdx = np.arange(ntmax)
kdx = np.arange(kmax)
ldx = np.arange(lmax)

xdx = np.arange(xmax)
pdx = np.arange(pmax)
pntdx = np.arange(pntmax)
rdx = np.arange(rmax)

#Export Dataframes to excel
data = 'WTM_Parameters_Empty.xlsx'
writer = pd.ExcelWriter(data, engine='xlsxwriter')
with writer as writer:
    Intermediate_Inputs.to_excel(writer, sheet_name='A', merge_cells=False)
    Final_Demand.to_excel(writer, sheet_name='y')
    Factor_Inputs.to_excel(writer, sheet_name='Fk', merge_cells=False)
    Factor_Input_Costs.to_excel(writer, sheet_name='Fl', merge_cells=False)
    Endowments.to_excel(writer, sheet_name='f')
    Factor_Prices.to_excel(writer, sheet_name='pi')
    Identity.to_excel(writer, sheet_name='I')
    Identity_RCOT.to_excel(writer, sheet_name='It')
writer.save()

#Import full dataframes back to python
data = pd.ExcelFile("WTM_Parameters.xlsx")
Intermediate_Inputs = data.parse(sheetname="A", header=0, parse_cols=(2+tmax), index_col=[0,1]).stack().reindex(index=Aindex).unstack()
Factor_Inputs = data.parse(sheetname="Fk", header=0, parse_cols=(2+tmax), index_col=[0,1]).stack().reindex(index=Fkindex).unstack()
Factor_Input_Costs = data.parse(sheetname="Fl", header=0, parse_cols=(2+tmax), index_col=[0,1]).stack().reindex(index=Flindex).unstack()
Final_Demand = data.parse(sheetname="y", header=0, parse_cols=(1+nmax)).stack().reindex(index=yindex).unstack()
Endowments = data.parse(sheetname="f", header=0, parse_cols=(1+kmax)).stack().reindex(index=findex).unstack()
Factor_Prices = data.parse(sheetname="pi", header=0, parse_cols=(1+lmax)).stack().reindex(index=piindex).unstack()
Identity_RCOT = data.parse(sheetname="It", header=0, parse=(1+tmax)).stack().reindex(index=Itindex).unstack()


#Build no-trade structures
#Intermediate_Inputs_NT = Intermediate_Inputs.loc(axis=0)[:,No_Trade_Sectors]
#Identity_RCOT_NT = Identity_RCOT.loc(axis=0)[No_Trade_Sectors]
#Final_Demand_NT = Final_Demand.transpose()
#Final_Demand_NT = Final_Demand_NT.loc(axis=0)[No_Trade_Sectors]
#data = 'NTM_Parameters.xlsx'
#writer = pd.ExcelWriter(data, engine='xlsxwriter')
#with writer as writer:
#    Intermediate_Inputs_NT.to_excel(writer, sheet_name='Ant', merge_cells=False)
#    Identity_RCOT_NT.to_excel(writer, sheet_name='Itnt', merge_cells=False)
#    Final_Demand_NT.to_excel(writer, sheet_name='ynt', merge_cells=False)
#writer.save()
#data = pd.ExcelFile("NTM_Parameters.xlsx")
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
It = Identity_RCOT.values
#Ant = Intermediate_Inputs_NT.to_panel().values.transpose(1,2,0)
#Itnt = Identity_RCOT_NT.values
#ynt = Final_Demand_NT.values

IA = It - A
#IAnt = Itnt - Ant


#Check numpy arrays for NaN
#np.isnan(np.sum(A))


#Flpiindex = pd.MultiIndex.from_product([Technologies, Regions], names=['Technologies','Regions'])
#Flpi = pd.Series(data='', index=Flpiindex).unstack().values

#for i in idx:
    #for t in tdx:
        #Flpi[t,i] = np.sum(Fl[t,l,i] * pi[l,i] for l in ldx)




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
#NTM.solve()
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
#NTM_Dual.solve()
NTM_Price.solve(pulp.CPLEX_PY())

#Results
Prices_NT = pd.Series('', index=pntindex)
for v in pntindex:
  Prices_NT[v] = p_nt[v].varValue 
Prices_NT = Prices_NT.unstack()
p_nt = Prices_NT.values




#PULP World Trade Model PRIMAL (Quantity Model)

#LP Define
WTM_Quantity = pulp.LpProblem("WTM_Quanity", pulp.LpMinimize)

#Variables
x = pulp.LpVariable.dicts("Production", xindex, 0)

#Objective Function
WTM_Quantity += np.sum([pi[i,l] * Fl[t,i,l] * x[Technologies[t], Regions[i]] for t in tdx for i in idx for l in ldx]) 

#Constraints 
for n in ndx:
    WTM_Quantity += np.sum([[IA[i, n, t] * x[Technologies[t],Regions[i]] for t in tdx] for i in idx]) >= np.sum([y[i,n] for i in idx])
    
for k in kdx:
    for i in idx:
        WTM_Quantity += np.sum([Fk[t, i, k] * x[Technologies[t],Regions[i]] for t in tdx]) <= f[i, k]

#p_nt = Prices_NT.values
#for i in idx:
#   WTM_Quantity += np.sum([p_nt[n,i] * IA[i, n, t] * x[Technologies[t],Regions[i]] for t in tdx for n in ndx]) <= np.sum([p_nt[n,i] * y[i,n] for n in ndx])

#for nt in ntdx:
#    for i in idx:
#       WTM_Quantity += np.sum([IAnt[i,nt,t] * x[Technologies[t],Regions[i]] for t in tdx]) >= ynt[nt,i]

#Solver
#WTM_Quantity.solve()
WTM_Quantity.solve(pulp.CPLEX_PY())

#Results
Production = pd.Series('', index=xindex)
for v in xindex:
  Production[v] = x[v].varValue 
Production = Production.unstack()
x = Production.values
Sum = Production.sum(axis=1)
Production = Production.assign(Total = Production.sum(axis=1))




#PULP World Trade Model DUAL (Price Model)

#LP Define
WTM_Price = pulp.LpProblem("WTM_Price", pulp.LpMaximize)

#Variables
p = pulp.LpVariable.dicts("Prices", pindex, 0)
r = pulp.LpVariable.dicts("Rents", rindex, 0)
#alpha = pulp.LpVariable.dicts("Rents", alphaindex, 0)
#ntr = pulp.LpVariable.dicts("Rents", ntrindex, 0)

#Objective Function
WTM_Price += np.sum([p[Sectors[n]] * y[i,n] for i in idx for n in ndx]) - np.sum([r[Factors_k[k], Regions[i]] * f[i,k] for k in kdx for i in idx]) 

#Constraints
for t in tdx:
    for i in idx:
        WTM_Price += np.sum([IA[i, n, t] * p[Sectors[n]] for n in ndx]) - np.sum([Fk[t,i,k] * r[Factors_k[k], Regions[i]] for k in kdx]) <= np.sum([Fl[t,i,l] * pi[i,l] for l in ldx])
    
#Solver
#WTM_Price.solve()
WTM_Price.solve(pulp.CPLEX_PY())

#Results
Prices = pd.Series('', index=pindex)
for v in pindex:
  Prices[v] = p[v].varValue
p = Prices.values 
Prices = Prices.to_frame(name='Prices')

Factor_Rents = pd.Series('', index=rindex)
for v in rindex:
  Factor_Rents[v] = r[v].varValue 
Factor_Rents = Factor_Rents.unstack().replace(to_replace='None',value=0).replace(to_replace='NaN',value=0)
r = Factor_Rents.values

#Benefit_of_Trade_Rents = pd.Series('', index=alphaindex)
#for v in alphaindex:
#  Benefit_of_Trade_Rents[v] = alpha[v].varValue 
#Benefit_of_Trade_Rents = Benefit_of_Trade_Rents.to_frame(name='BOT Rents')
#alpha = Benefit_of_Trade_Rents.values

#No_Trade_Rents = pd.Series('', index=ntrindex)
#for v in ntrindex:
#  No_Trade_Rents[v] = ntr[v].varValue 
#No_Trade_Rents = No_Trade_Rents.unstack() 
#ntr = No_Trade_Rents.values 


#Print status of all model runs
print "Status:", pulp.LpStatus[NTM_Quantity.status]
print pulp.value(NTM_Quantity.objective)       
print "Status:", pulp.LpStatus[NTM_Price.status]
print pulp.value(NTM_Price.objective)  
print "Status:", pulp.LpStatus[WTM_Quantity.status]  
print pulp.value(WTM_Quantity.objective) 
print "Status:", pulp.LpStatus[WTM_Price.status]
print pulp.value(WTM_Price.objective)




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
Factor_Use = Factor_Use.assign(Total = Factor_Use.sum(axis=1))
            
#Factor_Costs
Flpi = np.ones((lmax,imax,tmax))
for t in tdx:
    for i in idx:
        for l in ldx:
            Flpi[l,i,t]  = Fl[t,i,l]*pi[i,l]
Factor_Costs = pd.Panel(data=Flpi, items=Factors_l, major_axis=Regions, minor_axis=Technologies)        
Factor_Costs = Factor_Costs.to_frame()

#Intermediate_Production_Costs
Ap = np.ones((nmax,imax,tmax))
for t in tdx:
    for i in idx:
        for n in ndx:
            Ap[n,i,t]  = A[i,n,t]*p[n]
Intermediate_Costs = pd.Panel(data=Ap, items=Sectors, major_axis=Regions, minor_axis=Technologies)        
Intermediate_Costs = Intermediate_Costs.to_frame()


#Factor_Rent_Payments
Fkr = np.ones((kmax,imax,tmax))
for t in tdx:
    for i in idx:
        for k in kdx:
            Fkr[k,i,t]  = Fk[t,i,k]*r[k,i]
Factor_Rent_Payments = pd.Panel(data=Fkr, items=Factors_k, major_axis=Regions, minor_axis=Technologies)        
Factor_Rent_Payments = Factor_Rent_Payments.to_frame()

#Benefit_of_Trade_Payments
#IApntalpha = np.ones((nmax,imax,tmax))
#for t in tdx:
#    for i in idx:
#        for n in ndx:
#            IApntalpha[n,i,t]  = IA[i,n,t]*p_nt[n,i]*alpha[i]
#Benefit_of_Trade_Rent_Payments = pd.Panel(data=IApntalpha, items=Sectors, major_axis=Regions, minor_axis=Technologies)        
#Benefit_of_Trade_Rent_Payments = Benefit_of_Trade_Rent_Payments.to_frame()

#No_Trade_Rent_Payments
#IAntntr = np.ones((ntmax,imax,tmax))
#for t in tdx:
#    for i in idx:
#        for nt in ntdx:
#            IAntntr[nt,i,t]  = IAnt[i,nt,t]*ntr[nt,i]
#No_Trade_Rent_Payments = pd.Panel(data=IAntntr, items=No_Trade_Sectors, major_axis=Regions, minor_axis=Technologies)        
#No_Trade_Rent_Payments = No_Trade_Rent_Payments.to_frame()

#Pdec
Pdec = pd.concat([Factor_Costs, Intermediate_Costs, Factor_Rent_Payments], axis=1)
#Pdec = pd.concat([Factor_Costs, Intermediate_Costs, Factor_Rent_Payments, Benefit_of_Trade_Rent_Payments], axis=1)
#Pdec = pd.concat([Factor_Costs, Intermediate_Costs, Factor_Rent_Payments, Benefit_of_Trade_Rent_Payments, No_Trade_Rent_Payments], axis=1)
Totals = Production.stack().swaplevel(0,1).sortlevel('Regions')
Totals.name = 'Production'
Totals = Totals.to_frame()
Totals = Totals.assign(Pdec_Total = Pdec.sum(axis=1))
Totals = Totals.assign(Factor_Costs_Total = Factor_Costs.sum(axis=1))
Totals = Totals.assign(Intermediate_Costs_Total = Intermediate_Costs.sum(axis=1))
Totals = Totals.assign(Factor_Rent_Payments_Total = Factor_Rent_Payments.sum(axis=1))
#Totals = Totals.assign(Benefit_of_Trade_Rent_Payments_Total = Benefit_of_Trade_Rent_Payments.sum(axis=1))
#Totals = Totals.assign(No_Trade_Rent_Payments_Total = No_Trade_Rent_Payments.sum(axis=1))
#Pdec = pd.concat([Totals,Pdec],axis=1)

#Net Exports
Ax = np.ones((nmax,imax))
Ex = np.ones((nmax,imax))
Itx = np.dot(It,x)
for n in ndx:
    for i in idx:
        Ax[n,i] = np.sum([A[i,n,t] * x[t,i] for t in tdx])
        Ex[n,i] = Itx[n,i] - Ax[n,i] - y[i,n]
Net_Exports = pd.DataFrame(data=Ex, index=Sectors, columns=Regions)        




#Export results
data = 'WTM_Results.xlsx'
writer = pd.ExcelWriter(data, engine='xlsxwriter')
with writer as writer:
    Production_NT.to_excel(writer, sheet_name='x_nt', merge_cells=False)
    Prices_NT.to_excel(writer, sheet_name='p_nt', merge_cells=False)
    Production.to_excel(writer, sheet_name='x', merge_cells=False)
    Prices.to_excel(writer, sheet_name='p')
    Factor_Rents.to_excel(writer, sheet_name='r')
    #Benefit_of_Trade_Rents.to_excel(writer, sheet_name='alpha')
    #No_Trade_Rents.to_excel(writer, sheet_name='ntr')

    Factor_Use.to_excel(writer, sheet_name='Fkx', merge_cells=False)
    Net_Exports.to_excel(writer, sheet_name='Ex',merge_cells=False)
    
    Factor_Costs.to_excel(writer, sheet_name='Flpi',merge_cells=False)
    Intermediate_Costs.to_excel(writer, sheet_name='Ap', merge_cells=False)
    Factor_Rent_Payments.to_excel(writer, sheet_name='Fkr', merge_cells=False)
    #Benefit_of_Trade_Rent_Payments.to_excel(writer, sheet_name='Apntalpha', merge_cells=False)
    #No_Trade_Rent_Payments.to_excel(writer, sheet_name='Antntr', merge_cells=False)
    Totals.to_excel(writer, sheet_name='Pdec', merge_cells=False)
    
writer.save()

#Production.to_csv("Production.csv",sep=',') 
