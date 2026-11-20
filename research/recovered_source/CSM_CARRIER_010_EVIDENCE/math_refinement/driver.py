import check,os,json,shutil,pathlib
p=pathlib.Path(os.environ['TRAINLAB_OUTPUT_DIR']);shutil.copyfile('RESULTS.json',p/'RESULTS.json');r=json.load(open('RESULTS.json'));(p/'metrics.json').write_text(json.dumps({'cases':len(r['cases'])}))
