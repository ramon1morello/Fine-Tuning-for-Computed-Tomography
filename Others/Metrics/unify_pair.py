import os
import glob
import pandas as pd

def unify_metrics(directory, suffix):
	pattern = os.path.join(directory, f"metrics_*{suffix}.csv")
	files = glob.glob(pattern)
	dfs = []
	for file in files:
		base = os.path.basename(file)
		name = os.path.splitext(base)[0]
		df = pd.read_csv(file, sep=';')
		# Renomeia as colunas exceto 'PAR'
		df_ren = df.rename(columns={
			'PSNR': f'{name}_PSNR',
			'SSIM': f'{name}_SSIM',
			'PI': f'{name}_PI'
		})
		dfs.append(df_ren)
	# Faz merge progressivo pela coluna 'PAR'
	from functools import reduce
	df_merged = reduce(lambda left, right: pd.merge(left, right, on='PAR', how='outer'), dfs)
	df_merged = df_merged.sort_values('PAR').reset_index(drop=True)
	return df_merged

if __name__ == "__main__":
	metrics_dir = os.path.dirname(__file__)
	df_362px = unify_metrics(metrics_dir, '362px')
	df_240px = unify_metrics(metrics_dir, '240px')
	# Salva os resultados
	df_362px.to_csv(os.path.join(metrics_dir, 'unified_metrics_362px.csv'), sep=';', index=False)
	df_240px.to_csv(os.path.join(metrics_dir, 'unified_metrics_240px.csv'), sep=';', index=False)
	print('Arquivos unificados salvos como unified_metrics_362px.csv e unified_metrics_240px.csv')
