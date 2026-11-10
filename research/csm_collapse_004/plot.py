import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
arms=['hard','soft_routes','soft_admission','residual_front'];labels=['Unchanged CSM','Soft routing','Soft admission','Frontend residual'];colors=['#374151','#2563eb','#d97706','#059669']
fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
for arm,label,color in zip(arms,labels,colors):
 curves=[json.loads((p/f'seed-{s}'/f'{arm}.json').read_text())['curves'] for s in (11,23,37)];updates=[r['update'] for r in curves[0]]
 for ax,key in zip(axes,('nll','accuracy')):
  vals=np.array([[r['fit'][key] for r in c] for c in curves]);ax.plot(updates,vals.mean(0),marker='o',color=color,label=label);ax.fill_between(updates,vals.min(0),vals.max(0),color=color,alpha=.12)
axes[0].axhline(.7,color='grey',ls='--',lw=1);axes[0].set(title='Fitting loss',ylabel='NLL (lower is better)')
axes[1].axhline(.8,color='grey',ls='--',lw=1);axes[1].set(title='Fitting accuracy',ylabel='Correct next-byte predictions',ylim=(0,1));axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
for ax in axes:ax.set_xlabel('Optimizer updates');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
axes[0].legend(fontsize=9);fig.suptitle('CSM collapse diagnostic • one real 128-target excerpt • mean and range of 3 seeds\nTraining-fit evidence only; not a generalization benchmark',fontsize=12)
fig.savefig(p/'fit_diagnostics.png',dpi=180)
