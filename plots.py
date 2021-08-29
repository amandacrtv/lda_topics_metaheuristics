import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def evolutions_log_to_png():

   df_de = pd.read_csv('.\\log\\DE_log.csv', usecols=['Fevals', 'Best'])
   # df_de['Type'] = 'DE'
   df_de['Tipo'] = 'DE'
   # df_de = df_de.rename({'Fevals': '# Fitness Eval.', 'Best': 'Best Coherence'}, axis=1)
   df_de = df_de.rename({'Fevals': '# Fitness Eval.', 'Best': 'Melhor Coerência'}, axis=1)

   num = 20
   fevals = [num + 20*i for i in range(15)]

   df_ga = pd.read_csv('.\\log\\GA_log.csv', usecols=['Fevals', 'Best', 'Improvement'])

   keys_ga = df_ga['Fevals'].to_list()
   vals_ga = df_ga['Best'].to_list()

   df_ga = df_ga.iloc[0]

   f_val = df_ga['Best'] - df_ga['Improvement']

   keys_ga.insert(0, 1)
   vals_ga.insert(0, f_val)

   values = dict(zip(keys_ga, vals_ga))

   rows_ga = []
   for feval in fevals:
      currval = 0
      for i, key in enumerate(keys_ga):
         if i == len(keys_ga) - 1:
            currval = values[key]
            break
         if feval >= key and feval < keys_ga[i+1]:
            currval = values[key]
            break
      rows_ga.append(pd.Series({"Fevals": feval, "Best": currval}))

   df_ga = pd.DataFrame(rows_ga)

   # df_ga = df_ga.rename({'Fevals': '# Fitness Eval.', 'Best': 'Best Coherence'}, axis=1)
   df_ga = df_ga.rename({'Fevals': '# Fitness Eval.', 'Best': 'Melhor Coerência'}, axis=1)
   # df_ga['Type'] = 'GA'
   df_ga['Tipo'] = 'GA'


   df_gpso = pd.read_csv('.\\log\\GPSO_log.csv', usecols=['Fevals', 'gbest'])

   # df_gpso = df_gpso.rename({'Fevals': '# Fitness Eval.', 'gbest': 'Best Coherence'}, axis=1)

   # df_gpso['Type'] = 'GPSO'

   df_gpso = df_gpso.rename({'Fevals': '# Fitness Eval.', 'gbest': 'Melhor Coerência'}, axis=1)

   df_gpso['Tipo'] = 'GPSO'

   df_pso = pd.read_csv('.\\log\\PSO_log.csv', usecols=['Fevals', 'gbest'])

   # df_pso = df_pso.rename({'Fevals': '# Fitness Eval.', 'gbest': 'Best Coherence'}, axis=1)

   # df_pso['Type'] = 'PSO'

   df_pso = df_pso.rename({'Fevals': '# Fitness Eval.', 'gbest': 'Melhor Coerência'}, axis=1)

   df_pso['Tipo'] = 'PSO'

   df_sa = pd.read_csv('.\\log\\SA_log.csv', usecols=['Fevals', 'Best'])

   df_sa = df_sa.loc[df_sa['Fevals'].isin(fevals)]

   # df_sa = df_sa.rename({'Fevals': '# Fitness Eval.', 'Best': 'Best Coherence'}, axis=1)

   # df_sa['Type'] = 'SA'

   df_sa = df_sa.rename({'Fevals': '# Fitness Eval.', 'Best': 'Melhor Coerência'}, axis=1)

   df_sa['Tipo'] = 'SA'

   df_sade = pd.read_csv('.\\log\\saDE_log.csv', usecols=['Fevals', 'Best'])

   # df_sade = df_sade.rename({'Fevals': '# Fitness Eval.', 'Best': 'Best Coherence'}, axis=1)

   # df_sade['Type'] = 'saDE'

   df_sade = df_sade.rename({'Fevals': '# Fitness Eval.', 'Best': 'Melhor Coerência'}, axis=1)

   df_sade['Tipo'] = 'saDE'

   merged_df = pd.concat([df_pso, df_gpso, df_de, df_sade, df_ga, df_sa])

   # df_description = pd.read_csv('de_log_description.csv', usecols=['Fevals', 'Best'])

   # df_description['type'] = 'description'

   # df_readme = pd.read_csv('de_log_readme.csv', usecols=['Fevals', 'Best'])

   # df_readme['type'] = 'readme'

   # merged_df = pd.concat([df_description, df_readme])

   sns.set_theme(style="whitegrid", font_scale=2)

   a4_dims = (11.7, 8.27)
   fig, ax = plt.subplots(figsize=a4_dims)

   plot = sns.lineplot(
      # data=merged_df, y='Best Coherence', x='# Fitness Eval.', style='Type', hue='Type', markers=True, dashes=False, linewidth = 3
      data=merged_df, y='Melhor Coerência', x='# Fitness Eval.', style='Tipo', hue='Tipo', markers=True, dashes=False, linewidth = 3
   )
   # ax.set(xscale="log")
   # plt.grid(True, which="both")
   # ax.set(xticks=df.topic.values)
   # plt.legend([],[], frameon=False)
   ax.set_label('Type')
   plt.savefig('compare_evolutions_pt.svg', format="svg", bbox_inches='tight')


def evolutions_language():
   years = [2019, 2020]
   languages = ['jupyter_notebook', 'python']

   dfs = []

   for y in years:
      for l in languages:
         df = pd.read_csv(f'.\\log\\PSO_5stars_log_{l}_{y}.csv', usecols=['Fevals', 'gbest'])
         # df = df.rename({'Fevals': '# Fitness Eval.', 'gbest': 'Best Coherence'}, axis=1)
         df = df.rename({'Fevals': '# Fitness Eval.', 'gbest': 'Melhor Coerência'}, axis=1)
         # df['Year'] = y
         # df['Language'] = l.replace('_', ' ').title()
         df['Ano'] = y
         df['Linguagem'] = l.replace('_', ' ').title()
         dfs.append(df)
   
   dfs = pd.concat(dfs)
   
   sns.set_theme(style="whitegrid", font_scale=2)
   # muted= ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66", "#77BEDB"]
   a4_dims = (11.7, 8.27)
   fig, ax = plt.subplots(figsize=a4_dims)
   plot = sns.lineplot(
      # data=dfs, y='Best Coherence', x='# Fitness Eval.', style='Year', hue='Language', markers=False, dashes=True, palette="Set2", linewidth = 3)
      data=dfs, y='Melhor Coerência', x='# Fitness Eval.', style='Ano', hue='Linguagem', markers=False, dashes=True, palette="Set2", linewidth = 3)
   # plot = sns.relplot(data=dfs, y="Best Coherence", x="# Fitness Eval.", hue="Year", kind="line", col="Language", col_wrap=2)
   # ax.text(x=0.5, y=1.1, s='Coherence Values for each Language and Year', weight='bold', ha='center', va='bottom', transform=ax.transAxes)
   plt.savefig(f"rel_single_evolutions_lang_pt.svg", format="svg", bbox_inches='tight')

evolutions_log_to_png()