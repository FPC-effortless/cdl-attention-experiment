"""Render the observed training work without implying a held result."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from common import HERE, SEEDS, ARMS, sha256
from review.audit_training import checked_trace


def main():
    evidence=HERE/'evidence'
    audit=json.loads((evidence/'INCOMPLETE_CAMPAIGN_AUDIT.json').read_text())
    if audit['campaign_complete'] or audit['qualification_status']!='not_evaluated_incomplete_campaign':
        raise ValueError('expected an explicitly incomplete campaign audit')
    fits={(r['arm'],r['seed']):r for r in audit['runs']}
    retry=next(r for r in audit['incomplete_attempts'] if r['attempt']=='permitted_retry')
    colors={'continuous':'#196CA4','reset_training':'#6F8197','unresolved':'#CC7626'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.titleweight':'bold','axes.titlesize':12,
                         'figure.facecolor':'#F7F9FC','axes.facecolor':'#FFFFFF'})
    fig=plt.figure(figsize=(12.5,9.4))
    layout=fig.add_gridspec(2,2,height_ratios=[1,1.1],left=.17,right=.97,top=.83,bottom=.14,hspace=.51,wspace=.23)
    progress=fig.add_subplot(layout[0,:]);loss_axes=[fig.add_subplot(layout[1,index]) for index in range(2)]
    plotted=[]
    for seed in SEEDS:
        for arm in ARMS:
            fit=fits.get((arm,seed));is_complete=fit is not None
            name=('Continuous' if arm=='continuous' else 'Reset training')+f', seed {seed}'
            step=fit['steps'] if is_complete else retry['observed_updates']
            color=colors[arm if is_complete else 'unresolved']
            index=len(plotted)
            progress.barh(index,step,height=.62,color=color)
            progress.text(step+14,index,f'{step:,}' if is_complete else f'{step:,} — no endpoint',va='center',fontsize=9)
            path=evidence/'runs'/f'{arm}_seed{seed}'/'train.jsonl' if is_complete else HERE.parent/retry['trace_used']
            expected_hash=(audit['input_sha256'][str(path.relative_to(HERE.parent))]
                           if is_complete else retry['trace_sha256'])
            if sha256(path)!=expected_hash:
                raise ValueError('plot trace hash differs from observed audit: '+str(path))
            records,partial_bytes=checked_trace(path,updates=step,allow_partial=not is_complete)
            if len(records)!=step or (not is_complete and partial_bytes!=retry.get('trailing_partial_line_bytes',0)):
                raise ValueError('plot trace differs from observed audit: '+str(path))
            style={17:'-',29:'--',41:':'}[seed]
            for axis,key in zip(loss_axes,('semantic_loss','lm_loss')):
                window=50
                means=np.convolve([r[key] for r in records],np.ones(window)/window,mode='valid')
                axis.plot([r['step'] for r in records][window-1:],means,color=color,linestyle=style,
                          linewidth=1.7,alpha=.9 if not is_complete else .75,
                          label=name+(' (partial)' if not is_complete else ''))
            plotted.append(name)
    progress.set_yticks(range(len(plotted)),plotted);progress.invert_yaxis()
    progress.axvline(1200,color='#334B64',linewidth=1,linestyle='--')
    progress.set_xlim(0,1390);progress.set_xticks([0,300,600,900,1200])
    progress.set_xlabel('Logged updates in each retained attempt; fixed endpoint = 1,200')
    progress.set_title('Five completed endpoints; one unresolved replacement',loc='left',pad=14)
    progress.grid(axis='x',alpha=.15);progress.set_axisbelow(True)
    progress.legend(handles=[Patch(color=colors['continuous'],label='Completed continuous fit'),
                             Patch(color=colors['reset_training'],label='Completed reset-trained fit'),
                             Patch(color=colors['unresolved'],label='Unresolved replacement')],
                    loc='upper left',bbox_to_anchor=(0,1.33),ncol=3,frameon=False,fontsize=9)
    for axis,title,ylabel in zip(loss_axes,('Semantic loss on training batches','Language rehearsal on TRAIN windows'),
                                ('Mean loss over 50 updates','Mean NLL over 50 updates')):
        axis.set_title(title,loc='left',pad=13);axis.set_xlabel('Update');axis.set_ylabel(ylabel)
        axis.set_xlim(1,1200);axis.grid(alpha=.13);axis.axvline(100,color='#8B98A9',linewidth=.9,linestyle=':')
    handles,labels=loss_axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,loc='lower center',bbox_to_anchor=(.52,.052),fontsize=8.5,frameon=False)
    fig.suptitle('COG-FUSION-002 | Training campaign incomplete',x=.065,y=.985,ha='left',fontsize=20,fontweight='bold',color='#193750')
    fig.text(.065,.94,'Architecture unchanged: 997,674 active parameters. Held model performance has not been evaluated.',fontsize=11,color='#526378')
    fig.text(.065,.038,'Dotted line at update 100: supervised typed execution ends; actual generated answers are consumed throughout.',fontsize=8.5,color='#526378')
    fig.text(.065,.018,'Original interrupted attempt: 13 updates, retained separately. Partial attempts are not cumulative progress. Loss curves are training evidence only.',fontsize=8.5,color='#526378')
    target=evidence/'COG_FUSION_002_LEARNING.png';fig.savefig(target,dpi=180,facecolor=fig.get_facecolor());plt.close(fig)
    print(target)


if __name__=='__main__':
    main()
