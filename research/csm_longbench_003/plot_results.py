"""Render measured learning curves and intervention effects; no inferred data."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
arms=['instant','single','multi','multi_tb64'];labels=['Immediate','Single timescale','Multiple timescales','Multiple, 64-token gradient']
colors=['#374151','#2563eb','#059669','#d97706']
fig,ax=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
for arm,label,color in zip(arms,labels,colors):
 curves=[json.loads((p/f'seed-{s}'/f'{arm}-curve.json').read_text()) for s in (11,23,37)]
 updates=[r['update'] for r in curves[0]]
 values=np.array([[r['dev']['nll'] for r in c] for c in curves])
 assert all([r['update'] for r in c]==updates for c in curves)
 ax[0].plot(updates,values.mean(0),marker='o',label=label,color=color)
 ax[0].fill_between(updates,values.min(0),values.max(0),alpha=.1,color=color)
summary=json.loads((p/'summary.json').read_text())['arms'];x=np.arange(4)
for j,(key,label,color) in enumerate([('trace_reset','Clear trace','#2563eb'),('trace_swap','Swap trace','#059669'),('record_reset','Clear records','#d97706')]):
 ax[1].bar(x+(j-1)*.24,[summary[a][key+'_post256_delta'] for a in arms],width=.24,label=label,color=color)
ax[0].set(title='Learning curves: mean and seed range',xlabel='Optimizer updates',ylabel='DEV NLL (8,192 targets; lower is better)');ax[0].legend(fontsize=8)
ax[1].axhline(0,color='black',lw=.8);ax[1].set_xticks(x,['Immediate','Single','Multiple','Multiple\nTB64']);ax[1].set(title='Memory interventions after token 256',ylabel='NLL change (positive = intervention hurts)');ax[1].legend(fontsize=8)
for a in ax:a.spines[['top','right']].set_visible(False);a.grid(axis='y',alpha=.15)
fig.suptitle('CSM long-run diagnostics • 3 seeds • 262,144 training targets per fit',fontsize=13)
fig.savefig(p/'learning_diagnostics.png',dpi=180);fig.savefig(p/'learning_diagnostics.pdf')
