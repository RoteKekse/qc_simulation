import numpy as np
res_ppgd = []
res_dmrg = []
for i in range(0,27):
    name = "results/PPGD_CG_n2_"+str(round(0.8+0.05*i,4))+"_cc-pvtz_r16_f1_i30_results.csv"
    with open(name, "r") as file:         
        lines=file.readlines()
    line = lines[-1].split(',')
    res_ppgd.append(float(line[1]))
    name = "results/DMRG_n2_"+str(round(0.8+0.05*i,4))+"_cc-pvtz_r16_f_i15_results.csv"
    with open(name, "r") as file:            
         lines=file.readlines() 
    line = lines[-1].split(',')
    res_dmrg.append(float(line[1]))
    
print(res_ppgd)
print(res_dmrg)
res_ppgd = np.array(res_ppgd)
res_dmrg = np.array(res_dmrg)
print(res_ppgd-res_dmrg)
        

