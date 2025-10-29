from PIL import Image
from Pylette import extract_colors
from Pylette.src.palette import Palette
from ast import literal_eval as make_tuple
from coloraide import Color
from scenedetect import AdaptiveDetector, detect, FrameTimecode, open_video, save_images, split_video_ffmpeg, \
    video_splitter
from webcolors import rgb_to_hex
import ffmpeg
import numpy as np
import os
import pandas as pd
import shutil

print()
print('#*' * 24)
print('🔵 1. Parse the videos')
print('➡️ Get video info, split the video in scenes, save screenshots,\n'
      '➡️ extract colors and save them in a .pal.csv file.')
print('*#' * 24)

pd.options.mode.chained_assignment = None  # default='warn'
commercials_df: pd.DataFrame = pd.read_csv('initial_data/commercials_initial_metadata.csv')

counter_1: int = 0

# The number of colors to be extracted
palette_size: int = 32

for index, row in commercials_df.iterrows():
    commercial_id: str = row['commercial_id']
    video_path: str = f'videos/{commercial_id}.mp4'
    if not os.path.exists(video_path):
        continue
    os.makedirs(f'screenshots/{commercial_id}', exist_ok=True)
    # os.makedirs(f'thumbnails/{commercial_id}', exist_ok=True)
    video_tmp_path = f'screenshots/{commercial_id}/{commercial_id}.mp4'
    shutil.copyfile(video_path, video_tmp_path)
    print('-' * 48)
    print(f'📼 Video {(index + 1)}/{len(commercials_df)}:', commercial_id)

    video_info: dict = ffmpeg.probe(video_path)
    try:
        # `avg_frame_rate` is a fraction e.g. `25/1`
        avg_frame_rate: float = round(eval(video_info['streams'][0]['avg_frame_rate']), 2)
        # `display_aspect_ratio` is a ratio e.g. `16:9`
        aspect_ratio: float = round(eval(video_info['streams'][0]['display_aspect_ratio'].replace(':', '/')), 2)
    except:
        # `avg_frame_rate` is a fraction e.g. `25/1`
        avg_frame_rate: float = round(eval(video_info['streams'][1]['avg_frame_rate']), 2)
        # `display_aspect_ratio` is a ratio e.g. `16:9`
        aspect_ratio: float = round(eval(video_info['streams'][1]['display_aspect_ratio'].replace(':', '/')), 2)

    # Enrich commercials_df
    commercials_df.loc[index, 'avg_frame_rate'] = avg_frame_rate
    commercials_df.loc[index, 'aspect_ratio'] = aspect_ratio

    video_stream = open_video(video_tmp_path, backend='pyav')
    try:
        while True:
            frame: np.ndarray = video_stream.read()
            if frame is False:
                break
    except Exception as e:
        print(f'Error: {e}')
        continue

    frame_number: int = video_stream.frame_number

    scene_list: list[tuple[FrameTimecode, FrameTimecode]] = detect(
        video_path=video_tmp_path,
        # AdaptiveDetector handles fast camera movement better
        # ThresholdDetector handles fade out/fade in events.
        detector=AdaptiveDetector(),
        start_time=12,
        end_time=frame_number - 12,
        # If `start_in_scene` is True, len(scene_list) will always be >= 1
        start_in_scene=True,
    )

    split_video_ffmpeg(
        input_video_path=video_path,
        scene_list=scene_list,
        output_dir=f'screenshots/{commercial_id}',
        output_file_template='$VIDEO_NAME.ss$SCENE_NUMBER.mp4',
    )
    try:
        image_list: dict[int, list[str]] = save_images(
            scene_list=scene_list,
            video=video_stream,
            image_extension='png',
            output_dir=f'screenshots/{commercial_id}',
            image_name_template='$VIDEO_NAME.ss$SCENE_NUMBER',
            num_images=1,  # 1=the middle frame
        )
    except Exception as e:
        print(f'Error: {e}')
        continue

    video_splitter.is_mkvmerge_available()

    # find size (in # of frames) of each scene
    scene_size_list: list[int] = list()
    for scene in scene_list:
        scene_size_list.append(scene[1].frame_num - scene[0].frame_num)

    frame_tick_list: list[tuple[int, int]] = list()
    for scene in scene_list:
        frame_tick_list.append((scene[0].frame_num, scene[1].frame_num))

    palette_list: list[Palette] = list()
    for key in image_list:
        screenshot_path: str = f'screenshots/{commercial_id}/{image_list[key][0]}'
        palette: Palette = extract_colors(
            image=screenshot_path, palette_size=palette_size,
            resize=True,
            mode='KM',
        )
        palette_list.append(palette)

    ###########

    df_list: list = list()
    for idx, palette in enumerate(palette_list):
        df: pd.DataFrame = pd.DataFrame(index=range(palette_size))
        df['commercial_id'] = commercial_id
        df['scene'] = idx + 1
        df['scene_size'] = scene_size_list[idx]
        df['start_frame'] = frame_tick_list[idx][0]
        df['end_frame'] = frame_tick_list[idx][1]
        color_list = [rgb_to_hex(color.rgb) for color in palette.colors]
        df['hex_code'] = pd.Series(palette.colors).apply(lambda color: rgb_to_hex(color.rgb))
        df['frequency_within_the_scene'] = pd.Series(palette.frequencies)

        df_list.append(df)

    commercial_palette_df: pd.DataFrame = pd.concat(df_list)
    commercial_palette_df.reset_index(inplace=True, drop=True)
    os.makedirs(f'commercial_palettes', exist_ok=True)
    commercial_palette_df.to_csv(f'commercial_palettes/{commercial_id}.pal.csv', index=False)
    print(f'🎨 Saved `commercial_palettes/{commercial_id}.pal.csv`')
    # Remove unnecessary files:
    # - remove copy of video
    os.remove(video_tmp_path)
    # - remove video clips generated for screenshots
    for key, image_path_list in image_list.items():
        image_path: str = image_path_list[0]
        video_clip_path: str = image_path.replace('.png', '.mp4')
        os.remove(f'screenshots/{commercial_id}/{video_clip_path}')
    counter_1 += 1
    # Create or update an enriched file for commercials
    os.makedirs(f'general', exist_ok=True)
    commercials_df.to_csv(f'general/commercials.csv', index=False)
    print('-' * 48)
    print(f'📄 Updated `general/commercials.csv`')

######################

print()
print('#*' * 24)
print('🔵 2. Enrich the .pal.csv file with reference colors')
print('➡️ Assign the closest reference palette colors to each color extracted,\n'
      '➡️ remove the first and/or the last scene if `black` (from essential palette) is the predominant color,\n'
      '➡️ add for each color the frequency within the scene,\n'
      '➡️ update the .pal.csv file.')
print('*#' * 24)

commercials_df: pd.DataFrame = pd.read_csv('general/commercials.csv')
# essential_df = pd.read_csv('../colors/essential_palette.csv')
# extended_df = pd.read_csv('../colors/extended_palette.csv')
reference_palette_hierarchy_df = pd.read_csv('colors/reference_palette_hierarchy.csv')

extended_palette_colors_list: list[Color] = [
    Color('oklch', list(make_tuple(row['ext_oklch_coords'])))
    for _, row in reference_palette_hierarchy_df.iterrows()
]


def get_closest_color_from_ref_palettes(
        oklch_color: Color,
) -> list:
    closest_color: Color = oklch_color.closest(colors=extended_palette_colors_list)
    index: int = extended_palette_colors_list.index(closest_color)
    return [
        reference_palette_hierarchy_df.loc[index, 'ext_color_name'],
        reference_palette_hierarchy_df.loc[index, 'ess_color_name'],
        reference_palette_hierarchy_df.loc[index, 'bas_color_name'],
    ]


counter: int = 0
for index, row in commercials_df.iterrows():
    commercial_id: str = row['commercial_id']
    print('-' * 48)
    print(f'📼 Video {(index + 1)}/{len(commercials_df)}:', commercial_id)
    counter += 1
    pal_path: str = f'commercial_palettes/{commercial_id}.pal.csv'
    pal_df: pd.DataFrame = pd.read_csv(
        pal_path,
        # usecols=['commercial_id', 'scene', 'start_frame', 'end_frame', 'hex_code', 'frequency'],
        index_col=False,
    )
    # Drop rows with null values for the column `hex_code`
    pal_df.dropna(subset=['hex_code'], inplace=True)
    # Assign the closest color form the 3 reference palettes (ESSENTIAL, EXTENDED, BASIC)
    try:
        pal_df[['closest_color_ext_pal', 'closest_color_ess_pal', 'closest_color_bas_pal']] = pal_df.apply(
            lambda row: get_closest_color_from_ref_palettes(
                oklch_color=Color(row['hex_code']).convert(space='oklch')),
            axis=1,
            result_type='expand',
        )
    except Exception as e:
        print('❌ ' + commercial_id, e)
        print()
        continue
    ## Remove 1st and last scene if black (from essential palette) is the predominant color
    scene_count: int = len(pal_df['scene'].unique())
    ## Group colors by closest_color_ess_pal
    groupby_closest_color_ess_pal_df = pal_df.groupby(['closest_color_ess_pal', 'scene'], as_index=False)[
        'frequency_within_the_scene'].sum()
    ## Remove the 1st and/or the last scene (only if scene_count > 1) if black (from essential palette) is greater than 90%
    if scene_count > 1:
        # Get colors from first scene
        first_condition = (
                groupby_closest_color_ess_pal_df[
                    (groupby_closest_color_ess_pal_df['scene'] == 1) & (
                            groupby_closest_color_ess_pal_df['closest_color_ess_pal'] == 'black')][
                    'frequency_within_the_scene'] > 0.90)
        if len(first_condition) > 0 and first_condition.values[0] == True:
            pal_df.drop(pal_df[pal_df['scene'] == 1].index, inplace=True)
            # Remove related screenshot
            os.remove(f'screenshots/{commercial_id}/{commercial_id}.ss001.png')

        # Get colors from last scene.
        last_condition = (
                groupby_closest_color_ess_pal_df[(groupby_closest_color_ess_pal_df['scene'] == scene_count) & (
                        groupby_closest_color_ess_pal_df['closest_color_ess_pal'] == 'black')][
                    'frequency_within_the_scene'] > 0.90)
        if len(last_condition) > 0 and last_condition.values[0] == True:
            pal_df.drop(pal_df[pal_df['scene'] == scene_count].index, inplace=True)
            # Remove related screenshot.
            os.remove(f'screenshots/{commercial_id}/{commercial_id}.ss{str(scene_count).rjust(3, "0")}.png')

    pal_df.to_csv(pal_path, index=False)
    print(f'🎨 Updated `commercial_palettes/{commercial_id}.pal.csv`')

######################

print()
print('#*' * 24)
print('🔵 3. Create scenes.csv')
print('➡️ Create a new dataset just for the scenes of each video with their\n'
      '➡️ scene sizes, start frames, end frames, scene sizes normalized and scene duration in seconds')
print('*#' * 24)

commercials_df: pd.DataFrame = pd.read_csv(
    'general/commercials.csv'
)
pal_df_list: list[pd.DataFrame] = []
for index, row in commercials_df.iterrows():
    commercial_id: str = row['commercial_id']
    avg_frame_rate: float = row['avg_frame_rate']
    print('-' * 48)
    print(f'📼 Video {(index + 1)}/{len(commercials_df)}:', commercial_id)
    pal_df: pd.DataFrame = pd.read_csv(
        f'commercial_palettes/{commercial_id}.pal.csv',
        index_col=False, usecols=['commercial_id', 'scene', 'scene_size', 'start_frame', 'end_frame'],
    )
    pal_df.drop_duplicates(subset=['commercial_id', 'scene', 'scene_size'], inplace=True)
    pal_df['scene_size_norm'] = round(pal_df['scene_size'] / pal_df['scene_size'].sum(), 2)
    pal_df['scene_duration_in_seconds'] = pal_df['scene_size'] / avg_frame_rate
    pal_df_list.append(pal_df)
    # Add the duration of each commercial (after removing `black` scenes) to commercials.csv (`duration_in_seconds`)
    commercials_df.loc[index, 'duration_in_seconds'] = round(pal_df['scene_duration_in_seconds'].sum(), 2)
    commercials_df.to_csv(f'general/commercials.csv', index=False)
    print('-' * 48)
    print(f'📄 Updated `general/commercials.csv`')
commercials_with_scenes_info_df = pd.concat(pal_df_list)

commercials_with_scenes_info_df.to_csv(
    'general/scenes.csv',
    index=False,
)
print('-' * 48)
print(f'📄 Updated `general/scenes.csv`')

######################
print()
print('#*' * 24)
print('🔵 4. Generate scene thumbnails')
print('➡️ Convert the PNG screenshots to low resolution WEBP thumbnails (in the folder `thumbnails`),\n'
      '➡️ delete PNG screenshots and `screenshots` folder.')
print('*#' * 24)

pal_df_list: list[pd.DataFrame] = []
commercials_df: pd.DataFrame = pd.read_csv('general/commercials.csv')
scenes_df: pd.DataFrame = pd.read_csv('general/scenes.csv')

for idx, row in commercials_df.iterrows():
    commercial_id: str = row['commercial_id']
    print('-' * 48)
    print(f'📼 Video {(idx + 1)}/{len(commercials_df)}:', commercial_id)
    commercial_screenshots_path: str = f'screenshots/{commercial_id}'
    commercial_thumbnails_path: str = f'thumbnails/{commercial_id}'
    os.makedirs(f'thumbnails/{commercial_id}', exist_ok=True)
    single_commercial_scenes_df: pd.DataFrame = scenes_df[scenes_df['commercial_id'] == commercial_id].reset_index()
    for scenes_idx, scenes_row in single_commercial_scenes_df.iterrows():
        scene: str = scenes_row['scene']
        thumbnail_height: int = 180
        # Open the PNG screenshot
        screenshot_name_without_ext: str = f'{commercial_id}.ss{str(scene).rjust(3, "0")}'
        screenshot_image: Image = Image.open(f'{commercial_screenshots_path}/{screenshot_name_without_ext}.png')
        new_width: int = screenshot_image.size[0] * thumbnail_height // screenshot_image.size[1]
        screenshot_image = screenshot_image.resize(size=(new_width, thumbnail_height))
        rgb_screenshot_image: Image.Image = screenshot_image.convert('RGB')
        # Exporting the WEBP thumbnail
        rgb_screenshot_image.save(fp=f'{commercial_thumbnails_path}/{screenshot_name_without_ext}.webp', format='webp')
        print(f'🖼️ Scene {scenes_idx + 1}/{len(single_commercial_scenes_df)}: '
              f'`{screenshot_name_without_ext}.png` => `{screenshot_name_without_ext}.webp`')
    # Remove the commercial screenshot folder.
    shutil.rmtree(commercial_screenshots_path)
    print(f'🗑️ Deleted `{commercial_screenshots_path}` folder')

# Finally remove the screenshot folder.
shutil.rmtree('screenshots')
print('-' * 48)
print(f'🗑️ Deleted `screenshots` folder')

######################

print()
print('#*' * 24)
print('🔵 5. Enrich the .pal.csv file with the term frequency of each color in each palette')
print('➡️ Calculate the tf and update the .pal.csv file.')
print('*#' * 24)

commercials_df: pd.DataFrame = pd.read_csv('general/commercials.csv')
scene_sizes_df: pd.DataFrame = pd.read_csv(
    'general/scenes.csv',
    usecols=['commercial_id', 'scene', 'scene_size_norm'])

for index, row in commercials_df.iterrows():
    commercial_id: str = row['commercial_id']
    print('-' * 48)
    print(f'📼 Video {(index + 1)}/{len(commercials_df)}:', commercial_id)
    pal_df: pd.DataFrame = pd.read_csv(
        f'commercial_palettes/{commercial_id}.pal.csv',
        index_col=False,
    )
    # Drop rows with null values for the column `hex_code`
    pal_df.dropna(subset=['hex_code'], inplace=True)
    # Add scene_size column
    pal_df = pd.merge(pal_df, scene_sizes_df, how='inner', on=['commercial_id', 'scene'])
    ## Add frequency_within_the_commercial
    pal_df['frequency_within_the_commercial'] = pal_df['scene_size_norm'] * pal_df['frequency_within_the_scene']
    ## Calculate tf (term frequency) of each closest_color_ext_pal for each commercial
    ## Group colors by closest_color_ext_pal
    groupby_closest_color_ext_pal_df = pal_df.groupby(['closest_color_ext_pal'], as_index=False)[
        'frequency_within_the_commercial'].sum()
    for index_2, row_2 in groupby_closest_color_ext_pal_df.iterrows():
        pal_df.loc[pal_df['closest_color_ext_pal'] == row_2['closest_color_ext_pal'], 'tf'] = row_2[
            'frequency_within_the_commercial']
    pal_df.to_csv(f'commercial_palettes/{commercial_id}.pal.csv', index=False)
    print(f'🎨 Updated `commercial_palettes/{commercial_id}.pal.csv`')

######################
print()
print('#*' * 24)
print('🔵 6. Concatenate all commercial palettes in a unique CSV file')
print('➡️ Create the file commercial_palettes.csv file,\n'
      '➡️ delete the commercial_palettes folder.')
print('-' * 48)
print('*#' * 24)

# Concatenate all commercial palettes in a unique CSV file

commercials_df: pd.DataFrame = pd.read_csv('general/commercials.csv')

pal_df_list: list[pd.DataFrame] = []
for index, row in commercials_df.iterrows():
    commercial_id: str = row['commercial_id']
    print(f'📼 Video {(index + 1)}/{len(commercials_df)}:', commercial_id)
    try:
        palette_df: pd.DataFrame = pd.read_csv(
            f'commercial_palettes/{commercial_id}.pal.csv',
        )
    except Exception as e:
        print('❌ ' + commercial_id, e)
        continue
    pal_df_list.append(palette_df)
commercial_palettes_df = pd.concat(pal_df_list).reset_index(drop=True)

os.makedirs('colors', exist_ok=True)
# Save the file CSV
commercial_palettes_df.to_csv('colors/commercial_palettes.csv', index=False)
print('-' * 48)
print(f'🎨 Saved `colors/commercial_palettes.csv`')

# Finally remove the commercial_palettes folder
shutil.rmtree('commercial_palettes')
print('-' * 48)
print(f'🗑️ Deleted `commercial_palettes` folder')
