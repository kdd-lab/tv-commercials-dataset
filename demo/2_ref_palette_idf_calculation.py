import numpy as np
import os
import pandas as pd

print()
print('#*' * 24)
print('🔵 Calculate the idfs (Inverse Document Frequencies) of each color for each reference palette')
print('*#' * 24)
print('➡️ Calculate the idfs,\n'
      '➡️ save them in a CSV file (one for each reference palette).')
print('*#' * 24)

# Open the file with all the palettes
commercial_palettes_df: pd.DataFrame = pd.read_csv('colors/commercial_palettes.csv')
commercials_df: pd.DataFrame = pd.read_csv('general/commercials.csv')

# Count the number of documents in which every basic palette color appears

group_by_closest_color_bas_pal_df = (
    commercial_palettes_df.groupby(['closest_color_bas_pal', 'commercial_id'], as_index=False)
    .size().groupby(['closest_color_bas_pal'], as_index=False).size()
)

group_by_closest_color_bas_pal_df['idf'] = np.log10(len(commercials_df) / group_by_closest_color_bas_pal_df['size'])

os.makedirs(name='colors', exist_ok=True)
(group_by_closest_color_bas_pal_df
 .to_csv('colors/basic_palette_idfs.csv', index=False)
 )
print('-' * 48)
print(f'📄 Saved `colors/basic_palette_idfs.csv`')

# Count the number of documents in which every essential palette color appears

group_by_closest_color_ess_pal_df = (
    commercial_palettes_df.groupby(['closest_color_ess_pal', 'commercial_id'], as_index=False)
    .size().groupby(['closest_color_ess_pal'], as_index=False).size()
)

group_by_closest_color_ess_pal_df['idf'] = np.log10(len(commercials_df) / group_by_closest_color_ess_pal_df['size'])

(group_by_closest_color_ess_pal_df
 .to_csv('colors/essential_palette_idfs.csv', index=False)
 )
print('-' * 48)
print(f'📄 Saved `colors/essential_palette_idfs.csv`')

# Count the number of documents in which every extended palette color appears

group_by_closest_color_ext_pal_df = (
    commercial_palettes_df.groupby(['closest_color_ext_pal', 'commercial_id'], as_index=False)
    .size().groupby(['closest_color_ext_pal'], as_index=False).size()
)

group_by_closest_color_ext_pal_df['idf'] = np.log10(len(commercials_df) / group_by_closest_color_ext_pal_df['size'])
group_by_closest_color_ext_pal_df.drop(columns='size', inplace=True)
(group_by_closest_color_ext_pal_df
 .to_csv('colors/extended_palette_idfs.csv', index=False)
 )
print('-' * 48)
print(f'📄 Saved `colors/extended_palette_idfs.csv`')
