"""Render source-backed research figures after all qualification receipts exist."""
from pathlib import Path
import json
import math

from common import HERE, SEEDS
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    evidence=HERE/'evidence'
    q=json.loads((evidence/'QUALIFICATION.json').read_text())
    runtime=json.loads((evidence/'RUNTIME_RESULTS.json').read_text())
    language=json.loads((evidence/'LANGUAGE_PRESERVATION.json').read_text())
    colors={'parent':'#8895A7','reset_training':'#D4863B','continuous':'#196CA4'}
    labels={'parent':'Frozen parent','reset_training':'Reset during training','continuous':'Continuous training'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.titleweight':'bold','axes.titlesize':12,
                         'figure.facecolor':'#F7F9FC','axes.facecolor':'#FFFFFF'})
    figure,axes=plt.subplots(2,2,figsize=(12.8,9.1))
    x=np.arange(3);width=.24
    for ax,panel,title in ((axes[0,0],'test','Familiar wording: complete paired histories'),
                           (axes[0,1],'fresh','New wording: complete paired histories')):
        for j,arm in enumerate(('parent','reset_training','continuous')):
            counts=[q['runs'][f'{arm}_seed{s}']['panels'][panel] for s in SEEDS]
            values=[100*c['whole_session_pairs_exact']/c['families'] for c in counts]
            bars=ax.bar(x+(j-1)*width,values,width,color=colors[arm],label=labels[arm])
            for bar,count in zip(bars,counts):
                ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1.3,
                        str(count['whole_session_pairs_exact']),ha='center',va='bottom',fontsize=8)
        ax.axhline(95,color='#526378',ls='--',lw=1)
        ax.set_ylim(0,112);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Complete pairs (%)')
        ax.set_xticks(x,[f'Seed {s}' for s in SEEDS]);ax.set_title(title,loc='left',pad=14)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        ax.text(.01,-.16,'Bar labels: correct pairs out of 128. All held runs carry state.',
                transform=ax.transAxes,fontsize=8.5,color='#526378')
    ax=axes[1,0]
    original=language['references']['original_connected_cog']['mean_nll']
    for j,arm in enumerate(('parent','reset_training','continuous')):
        if arm=='parent':
            values=[language['references']['paired_fused001_parents'][str(s)]['metrics']['mean_nll'] for s in SEEDS]
        else:
            values=[next(r['metrics']['mean_nll'] for r in language['runs'] if r['arm']==arm and r['seed']==s) for s in SEEDS]
        ax.scatter(x+(j-1)*.13,values,s=68,color=colors[arm],zorder=3,edgecolors='white',linewidths=.6)
        for xi,value in zip(x+(j-1)*.13,values):
            ax.annotate(f'{value:.3f}',(xi,value),xytext=(0,7 if arm!='reset_training' else -15),
                        textcoords='offset points',ha='center',fontsize=8)
    ax.axhline(original,color='#394C62',lw=1.5,label='Original full COG')
    ax.axhline(original*1.1,color='#526378',ls='--',lw=1,label='Original +10% NLL')
    ax.set_xticks(x,[f'Seed {s}' for s in SEEDS]);ax.set_ylabel('NLL, nats per character (lower is better)')
    ax.set_title('Ordinary language: original full reference retained',loc='left',pad=14)
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.legend(fontsize=8,loc='best',frameon=False)
    ax.text(.01,-.16,'Same 8,192 character targets; previously exposed regression panel.',
            transform=ax.transAxes,fontsize=8.5,color='#526378')
    ax=axes[1,1]
    for j,arm in enumerate(('parent','reset_training','continuous')):
        name='parent_001' if arm=='parent' else arm
        rows=[next(r['new_schedule'] for r in runtime['runs'] if r['arm']==name and r['seed']==s) for s in SEEDS]
        values=[100*r['learned_questions_correct']/r['learned_question_count'] for r in rows]
        bars=ax.bar(x+(j-1)*width,values,width,color=colors[arm])
        for bar,row in zip(bars,rows):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1.3,
                    f"{row['learned_questions_correct']}/{row['learned_question_count']}",
                    ha='center',va='bottom',fontsize=8)
    ax.set_ylim(0,112);ax.set_yticks([0,25,50,75,100]);ax.set_xticks(x,[f'Seed {s}' for s in SEEDS])
    ax.set_ylabel('Exact generated answers (%)');ax.set_title('Actual sessions: shifted question timing',loc='left',pad=14)
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax.text(.01,-.16,'Predeclared runtime scenario; descriptive diagnostic, not a broad benchmark.',
            transform=ax.transAxes,fontsize=8.5,color='#526378')
    handles,legend=axes[0,0].get_legend_handles_labels()
    figure.legend(handles,legend,loc='upper center',bbox_to_anchor=(.5,.935),ncol=3,frameon=False,fontsize=10)
    figure.suptitle('COG-FUSION-002 | Learning across generated answers',x=.07,ha='left',y=.989,fontsize=20,fontweight='bold',color='#193750')
    figure.text(.07,.947,'Same 997,674 active parameters. Three matched seeds. Fixed 1,200-update endpoints.',fontsize=11,color='#526378')
    figure.text(.07,.018,'Source: saved model outputs, independently reconciled qualification, runtime and language receipts. Seeds reuse the same held families.',fontsize=8.8,color='#526378')
    figure.subplots_adjust(left=.08,right=.97,top=.86,bottom=.11,wspace=.27,hspace=.56)
    target=evidence/'COG_FUSION_002_RESULTS.png'
    figure.savefig(target,dpi=180,facecolor=figure.get_facecolor());plt.close(figure)

    figure,axes=plt.subplots(1,2,figsize=(12.5,4.8))
    for arm in ('continuous','reset_training'):
        for seed in SEEDS:
            rows=[json.loads(line) for line in (evidence/'runs'/f'{arm}_seed{seed}'/'train.jsonl').read_text().splitlines()]
            steps=np.array([r['step'] for r in rows])
            for ax,key in zip(axes,('semantic_loss','lm_loss')):
                values=np.array([r[key] for r in rows]);window=50
                mean=np.convolve(values,np.ones(window)/window,mode='valid')
                ax.plot(steps[window-1:],mean,color=colors[arm],alpha=.65,
                        linestyle={17:'-',29:'--',41:':'}[seed],label=f'{labels[arm]}, seed {seed}')
                ax.axvline(100,color='#69788D',lw=.8,ls=':');ax.grid(alpha=.13)
                ax.set_xlabel('Update');ax.set_xlim(1,1200)
    axes[0].set_title('Semantic loss on training batches',loc='left');axes[0].set_ylabel('Mean loss over 50 updates')
    axes[1].set_title('Language rehearsal loss on TRAIN windows',loc='left');axes[1].set_ylabel('Mean NLL over 50 updates')
    axes[1].legend(frameon=False,fontsize=8,ncol=2,loc='upper right')
    figure.suptitle('Learning traces | fixed endpoints, no checkpoint selection',x=.07,ha='left',fontsize=16,fontweight='bold')
    figure.text(.07,.015,'These are training losses, not held validation scores. Dotted vertical line marks the end of gold typed execution; generated answers are used throughout.',fontsize=8.5,color='#526378')
    figure.subplots_adjust(left=.07,right=.98,top=.83,bottom=.17,wspace=.23)
    figure.savefig(evidence/'COG_FUSION_002_LEARNING.png',dpi=180,facecolor=figure.get_facecolor());plt.close(figure)
    print(target)


if __name__=='__main__':
    main()
