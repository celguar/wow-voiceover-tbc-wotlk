from wowvo_client.consts import race_gender_tuple_to_strings
from wowvo_client import utils
import numpy as np
import pandas as pd
from corrections.corrections import recode_expansions, apply_corrections, add_new_entries
from data_prep.sql_queries import query_dataframe_for_all_quests_and_gossip, query_dataframe_for_turtle, is_vanilla_broadcast_text, is_tbc_broadcast_text, is_wotlk_broadcast_text


def clean_quest_data(tts_processor):

    tts_processor.set_module_name('AI_VoiceOverData_TBC')
    tts_processor.load_problematic_quests_cache(tts_processor.problematic_quest_path)
    tts_processor.load_problematic_gossip_cache(tts_processor.problematic_gossip_path)

    language_code = "enUS"
    language_number = utils.language_code_to_language_number(language_code)
    print(f"Selected language: {language_code}")

    #Load data from our SQL query with tbc/wotlk data
    df = query_dataframe_for_all_quests_and_gossip(language_number)
    print("Data query OK.", flush = True)
    print(f"Number of rows: " + str('{:,}'.format(df.shape[0])))

    #mark custom models with -77 for sex and gender so code doesn't break when hashing
    mask = df['DisplayRaceID'].isna() & df['DisplaySexID'].isna()
    df.loc[mask, ['DisplayRaceID', 'DisplaySexID']] = -77

    #caterogize quests and gossip by expansion...roughly since there is no clear cutoff

    #create expansion category
    # conditions = [
    #     (df['id'] >= 1) & (df['id'] < 15278),
    #     (df['id'] >= 15278) & (df['id'] < 23728),
    #     (df['id'] >= 23728) & (df['id'] < 80000)
    # ]

    # Values for each condition
    choices = [0, 1, 2]  # 0: Vanilla, 1: TBC, 2: WotLK

    # df['expansion'] = np.select(conditions, choices, default=-1)

    # conditions_npc = [
    #     (df['id'] > 0) & (df['expansion'] == 0),
    #     (df['id'] > 0) & (df['expansion'] == 1),
    #     (df['id'] > 0) & (df['expansion'] == 2)
    # ]
    # Mask to identify rows where expansion == -1
    mask = df['expansion'] == -1

    # Subset the DataFrame
    df_subset = df[mask]

    conditions_creature_gobject = [
        (df_subset['id'] > 0) & (df_subset['expansion'] == 0),
        (df_subset['id'] > 0) & (df_subset['expansion'] == 1),
        (df_subset['id'] > 0) & (df_subset['expansion'] == 2)
    ]

    new_expansion_values = np.select(conditions_creature_gobject, choices, default=-1)

    # df_subset['expansion'] = np.select(conditions_npc, choices, default=-1)

    # Conditions for just that subset
    # conditions2 = [
    #     (df_subset['id'] >= 89931) & (df_subset['id'] < 180918),
    #     (df_subset['id'] >= 180918) & (df_subset['id'] < 186585),
    #     (df_subset['id'] >= 186585)
    # ]

    # Apply np.select just to the subset
    # new_expansion_values = np.select(conditions2, choices, default=-1)

    # Assign result back to the original DataFrame
    df.loc[mask, 'expansion'] = new_expansion_values

    print(f"Number of rows: " + str('{:,}'.format(df.shape[0])))

    # recategorize expansion from corrections file before filtering, this will eliminate some vanilla npcs that are being
    # wrongly categorized as tbc in the logic above
    #do corrections
    df = recode_expansions(df, "corrections/corrections.xlsx")
    print("recoding expansions for NPCs and quests ...", flush = True)

    '''
        filter out vanilla observations to avoid overlap with original VO
        we know that original VO only missed custom models and item/gameobject quests
        filtering the rest out should do the trick
        We are specifically keeping troll males from vanilla to add that in since they
        weren't voiced in the original VO
    '''

    df = df[
        (df['expansion'] >= 0) |
        (df['type'].isin(['item','gameobject'])) |
        (df['DisplayRaceID'] == -77) |
        ((df['DisplayRaceID'] == 8) & (df['DisplaySexID'] == 0) |
         df['source'].isin(['progress']))
    ]

    print(f"Number of rows: " + str('{:,}'.format(df.shape[0])))
    #add missing quest entries from excel; note all columns must be filled out

    # df = add_new_entries(df,"corrections/new_entries.xlsx")
    # print("Applying new entries ...", flush = True)

    # check and mark rows that have different lines in vanilla-tbc-wotlk
    


    #add the new columns
    df = tts_processor.preprocess_dataframe(df)


    #do corrections
    df = apply_corrections(df, "corrections/corrections.xlsx")
    print("Applying corrections ...", flush = True)


    # Define the mapping of old values to new values
    replace_map = {
        'scourge_male': 'forsaken_male',
        'scourge_female': 'forsaken_female',
        'icetroll_male': 'troll_male',
        'narrator_male': 'narrator',
        'taunka_male': 'tauren_male',
        'naga_female':'demon_female'
    }

    # Apply the replacement
    df['voice_name'] = df['voice_name'].replace(replace_map)

    df['db_source'] = 'AzerothCore'

    #correct some gossips that have nan for quest and quest_title
    df['quest'] = df.quest.fillna('')
    df['quest_title'] = df.quest_title.fillna('')

    # Get unique race-gender combinations
    unique_race_gender_combos = df[[
        'DisplayRaceID', 'DisplaySexID']].drop_duplicates().values
    # Convert the unique race-gender combinations to a tuple
    race_gender_tuple = tuple(map(tuple, unique_race_gender_combos))


    selected_voice_names = race_gender_tuple_to_strings(
            race_gender_tuple)

    # Add the new values if not already in the list
    for new_value in replace_map.values():
        if new_value not in selected_voice_names:
            selected_voice_names.append(new_value)
    #append other race-gender combos that were not in the list for some reason
    selected_voice_names.append('ogre_male')
    selected_voice_names.append('rexxar')
    selected_voice_names.append('tirion')
    selected_voice_names.append('jaina')
    selected_voice_names.append('mechanical')
    selected_voice_names.append('demon_male')
    selected_voice_names.append('big_creature')
    selected_voice_names.append('varian')
    selected_voice_names.append('khadgar')
    selected_voice_names.append('arthas')
    selected_voice_names.append('thrall')
    selected_voice_names.append('boy')
    selected_voice_names.append('girl')
    selected_voice_names.append('sylvanas')
    selected_voice_names.append('furbolg_male')
    selected_voice_names.append('arakkoa_male')
    selected_voice_names.append('ancient')
    selected_voice_names.append('satyr')
    selected_voice_names.append('dryad')
    selected_voice_names.append('naaru')
    selected_voice_names.append('tuskarr_male')
    selected_voice_names.append('wolvar_male')
    selected_voice_names.append('gorloc_male')
    selected_voice_names.append('keeper')
    selected_voice_names.append('lich_king')
    selected_voice_names.append('human_necro')
    selected_voice_names.append('ogrila_ogre')
    selected_voice_names.append('titan_male')
    selected_voice_names.append('earthen')
    selected_voice_names.append('giant_male')
    selected_voice_names.append('dragon_male')
    selected_voice_names.append('arthas_dk')
    selected_voice_names.append('draenei_female_dk')
    selected_voice_names.append('draenei_male_dk')
    selected_voice_names.append('troll_male_dk')
    selected_voice_names.append('bloodelf_female_dk')
    selected_voice_names.append('bloodelf_male_dk')
    selected_voice_names.append('draenei_female_dk')
    selected_voice_names.append('orc_male_dk')
    selected_voice_names.append('nightelf_female_dk')
    selected_voice_names.append('forsaken_male_dk')
    selected_voice_names.append('gnome_female_dk')
    selected_voice_names.append('human_male_dk')
    selected_voice_names.append('human_necro_dk')


    del df_subset
    msg = "Quest and gossip data cleaning complete. Number of rows: " + str('{:,}'.format(df.shape[0]))

    return msg, df

def clean_quest_data_turtle(tts_processor):

    # # add turtle module
    # MODULE_NAMES = ['AI_VoiceOverData_Vanilla', 'AI_VoiceOverData_VanillaExtra', 'AI_VoiceOverData_TBC',
    #                   'AI_VoiceOverData_WoTLK', 'AI_VoiceOverData_Turtle']
    
    tts_processor.set_module_name('AI_VoiceOverData_Turtle')
    tts_processor.load_problematic_quests_cache(tts_processor.problematic_quest_path)
    tts_processor.load_problematic_gossip_cache(tts_processor.problematic_gossip_path)
    
    
    
    language_code = "enUS"
    language_number = utils.language_code_to_language_number(language_code)
    print(f"Selected language: {language_code}")

    #Load data from our SQL query with tbc/wotlk data
    df = query_dataframe_for_turtle(language_number)
    print("Data query OK.", flush = True)
    print(f"Number of rows: " + str('{:,}'.format(df.shape[0])))

    #mark custom models with -77 for sex and gender so code doesn't break when hashing
    mask = df['DisplayRaceID'].isna() & df['DisplaySexID'].isna()
    df.loc[mask, ['DisplayRaceID', 'DisplaySexID']] = -77

    #caterogize quests and gossip by expansion...roughly since there is no clear cutoff

    #create expansion category
    # conditions = [
    #     (df['id'] >= 1) & (df['id'] < 15278),
    #     (df['id'] >= 15278) & (df['id'] < 23728),
    #     (df['id'] >= 23728) & (df['id'] < 80000)
    # ]

    # Values for each condition
    choices = [1]  # 0: Vanilla, 1: TBC, 2: WotLK

    # df['expansion'] = np.select(conditions, choices, default=-1)

    # conditions_npc = [
    #     (df['id'] > 0) & (df['expansion'] == 0),
    #     (df['id'] > 0) & (df['expansion'] == 1),
    #     (df['id'] > 0) & (df['expansion'] == 2)
    # ]
    # Mask to identify rows where expansion == -1
    mask = df['expansion'] == -1
    # Subset the DataFrame
    df_subset = df[mask]
    conditions_creature_gobject = [
        (df_subset['id'] > 0) & (df_subset['expansion'] == 1)
    ]
    new_expansion_values = np.select(conditions_creature_gobject, choices, default=-1)

    # df_subset['expansion'] = np.select(conditions_npc, choices, default=-1)

    # Conditions for just that subset
    # conditions2 = [
    #     (df_subset['id'] >= 89931) & (df_subset['id'] < 180918),
    #     (df_subset['id'] >= 180918) & (df_subset['id'] < 186585),
    #     (df_subset['id'] >= 186585)
    # ]

    # Apply np.select just to the subset
    # new_expansion_values = np.select(conditions2, choices, default=-1)

    # Assign result back to the original DataFrame
    df.loc[mask, 'expansion'] = new_expansion_values

    def process_expansion_column(df):
        df = df.copy()
        
        def get_expansion(row):
            if row.get('source') != 'gossip':
                return row.get('expansion')
            
            broadcast_id = row.get('broadcast_text_id', 0)

            if not broadcast_id and not row['original_text']:
                return row.get('expansion')
            
            is_vanilla = is_vanilla_broadcast_text(broadcast_id, row['original_text'])
            is_turtle = not is_vanilla
            
            if is_turtle:
                return 1
            
            return row.get('expansion')
        
        df['expansion'] = df.apply(get_expansion, axis=1)
        return df

    df = process_expansion_column(df)

    df['zone'] = df['zone'].fillna('Unknown')
    mask = df['zone'] == 'Unknown'

    # Create a mapping from id to zone from accept rows (non-Unknown)
    accept_zones = (df[(df['source'] == 'accept') & (df['zone'] != 'Unknown')]
                .drop_duplicates(subset=['id'], keep='first')
                .set_index('id')['zone'])
    df.loc[mask, 'zone'] = df.loc[mask, 'id'].map(accept_zones)

    complete_zones = (df[(df['source'] == 'complete') & (df['zone'] != 'Unknown')]
                .drop_duplicates(subset=['id'], keep='first')
                .set_index('id')['zone'])
    df.loc[mask, 'zone'] = df.loc[mask, 'id'].map(complete_zones)

    # print(f"Number of rows: " + str('{:,}'.format(df.shape[0])))

    # recategorize expansion from corrections file before filtering, this will eliminate some vanilla npcs that are being
    # wrongly categorized as tbc in the logic above
    #do corrections
    # df = recode_expansions(df, "corrections/corrections.xlsx")
    # print("recoding expansions for NPCs and quests ...", flush = True)

    '''
        filter out vanilla observations to avoid overlap with original VO
        we know that original VO only missed custom models and item/gameobject quests
        filtering the rest out should do the trick
        We are specifically keeping troll males from vanilla to add that in since they
        weren't voiced in the original VO
    '''

    # Use tuple of (id, quest) as the identifier
    turtle_npcs = set()
    for _, row in df.iterrows():
        if row['expansion'] > 0:
            turtle_npcs.add(row['id'])

    turtle_quests = set()
    for _, row in df.iterrows():
        if row['expansion'] > 0:
            turtle_quests.add(row['quest'])

    df_turtle = df[
        (df['expansion'] > 0)
        ]
    
    num_turtle_only = df_turtle.shape[0]
    print(f"Number of rows TURTLE: " + str('{:,}'.format(num_turtle_only)))

    mask = (
        df['id'].isin(turtle_npcs) |
        df['quest'].isin(turtle_quests) |
        (df['expansion'] > 0)
    )

    not_deprecated = ~df['quest_title'].astype(str).str.contains(r'\[DEPRECATED\]|\[CANCELLED\]', na=False)

    df = df[mask & not_deprecated]

    num_turtle_linked = df.shape[0]
    
    print(f"With added Vanilla linked NPCs: " + str('{:,}'.format(num_turtle_linked)))

    # # Create mask for Turtle content
    # turtle_mask = (
    #     (df['expansion'] > 0)
    # )

    # # Create composite key
    # df['_key'] = df['id'].astype(str) + '|' + df['quest'].astype(str)

    # # Get keys that have Turtle content
    # turtle_keys = set(df.loc[turtle_mask, '_key'])

    # # Final filter
    # final_mask = (df['_key'].isin(turtle_keys)) | turtle_mask
    # df = df[final_mask].drop(columns=['_key'])

    # df = df[
    #     (df['expansion'] > 0) |
    #     (df['type'].isin(['item','gameobject'])) |
    #     (df['DisplayRaceID'] == -77) |
    #     ((df['DisplayRaceID'] == 8) & (df['DisplaySexID'] == 0) |
    #      df['source'].isin(['progress']))
    # ]

    # print(f"Number of rows: " + str('{:,}'.format(df.shape[0])))
    #add missing quest entries from excel; note all columns must be filled out

    # df = add_new_entries(df,"corrections/new_entries.xlsx")
    # print("Applying new entries ...", flush = True)


    #add the new columns
    df = tts_processor.preprocess_dataframe(df)


    #do corrections
    # df = apply_corrections(df, "corrections/corrections.xlsx")
    # print("Applying corrections ...", flush = True)


    # Define the mapping of old values to new values
    replace_map = {
        'scourge_male': 'forsaken_male',
        'scourge_female': 'forsaken_female',
        'icetroll_male': 'troll_male',
        'narrator_male': 'narrator',
        'taunka_male': 'tauren_male',
        'naga_female':'demon_female'
    }

    # Apply the replacement
    df['voice_name'] = df['voice_name'].replace(replace_map)

    df['db_source'] = 'TurtleWoW'
    df.loc[df['expansion'] == 1, 'expansion'] = 3

    #correct some gossips that have nan for quest and quest_title
    df['quest'] = df.quest.fillna('')
    df['quest_title'] = df.quest_title.fillna('')

    # Get unique race-gender combinations
    unique_race_gender_combos = df[[
        'DisplayRaceID', 'DisplaySexID']].drop_duplicates().values
    # Convert the unique race-gender combinations to a tuple
    race_gender_tuple = tuple(map(tuple, unique_race_gender_combos))


    selected_voice_names = race_gender_tuple_to_strings(
            race_gender_tuple)

    # Add the new values if not already in the list
    for new_value in replace_map.values():
        if new_value not in selected_voice_names:
            selected_voice_names.append(new_value)
    #append other race-gender combos that were not in the list for some reason
    selected_voice_names.append('ogre_male')
    selected_voice_names.append('rexxar')
    selected_voice_names.append('tirion')
    selected_voice_names.append('jaina')
    selected_voice_names.append('mechanical')
    selected_voice_names.append('demon_male')
    selected_voice_names.append('big_creature')
    selected_voice_names.append('varian')
    selected_voice_names.append('khadgar')
    selected_voice_names.append('arthas')
    selected_voice_names.append('thrall')
    selected_voice_names.append('boy')
    selected_voice_names.append('girl')
    selected_voice_names.append('sylvanas')
    selected_voice_names.append('furbolg_male')
    selected_voice_names.append('arakkoa_male')
    selected_voice_names.append('ancient')
    selected_voice_names.append('satyr')
    selected_voice_names.append('dryad')
    selected_voice_names.append('naaru')
    selected_voice_names.append('tuskarr_male')
    selected_voice_names.append('wolvar_male')
    selected_voice_names.append('gorloc_male')
    selected_voice_names.append('keeper')
    selected_voice_names.append('lich_king')
    selected_voice_names.append('human_necro')
    selected_voice_names.append('ogrila_ogre')
    selected_voice_names.append('titan_male')
    selected_voice_names.append('earthen')
    selected_voice_names.append('giant_male')
    selected_voice_names.append('dragon_male')
    selected_voice_names.append('arthas_dk')
    selected_voice_names.append('draenei_female_dk')
    selected_voice_names.append('draenei_male_dk')
    selected_voice_names.append('troll_male_dk')
    selected_voice_names.append('bloodelf_female_dk')
    selected_voice_names.append('bloodelf_male_dk')
    selected_voice_names.append('draenei_female_dk')
    selected_voice_names.append('orc_male_dk')
    selected_voice_names.append('nightelf_female_dk')
    selected_voice_names.append('forsaken_male_dk')
    selected_voice_names.append('gnome_female_dk')
    selected_voice_names.append('human_male_dk')
    selected_voice_names.append('human_necro_dk')


    del df_subset
    msg = "Quest and gossip data loaded. Turtle rows: " + str('{:,}'.format(num_turtle_only)) + ", Linked rows: " + str('{:,}'.format(num_turtle_linked))

    import gradio as gr
    return msg, df, gr.update(choices=
        ['AI_VoiceOverData_Turtle', 'AI_VoiceOverData_VanillaExtra',
        'Testing'],
        value='AI_VoiceOverData_Turtle') if tts_processor.get_module_name() == 'AI_VoiceOverData_Turtle' else gr.skip(), gr.update(choices=
                                ['AI_VoiceOverData_Turtle', 'AI_VoiceOverData_VanillaExtra',
                                'Testing'],
                                value='AI_VoiceOverData_Turtle') if tts_processor.get_module_name() == 'AI_VoiceOverData_Turtle' else gr.skip()
