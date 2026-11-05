import pymysql
import pandas as pd
from data_prep.env_vars import MYSQL_HOST, MYSQL_PORT, MYSQL_PASSWORD, MYSQL_USER, MYSQL_DATABASE

_db_connection = None
_db_connection_vmangos = None
_db_connection_tbc_cmangos = None
_db_connection_turtle = None

TURTLE_DB_NAME = "tw_world_new"
def make_connection():
    global _db_connection
    
    # Если коннекта еще нет, создаем его
    if _db_connection is None:
        _db_connection = pymysql.connect(
            host=MYSQL_HOST,
            port=int(MYSQL_PORT),
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE
        )
    else:
        # Проверяем, не отвалился ли существующий коннект (на всякий случай)
        try:
            _db_connection.ping(reconnect=True)
        except Exception:
            # Если отвалился — ping() с флагом reconnect=True поднимет его сам,
            # но если совсем всё плохо, пересоздаем вручную
            _db_connection = pymysql.connect(
                host=MYSQL_HOST,
                port=int(MYSQL_PORT),
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                database=MYSQL_DATABASE
            )
            
    return _db_connection    

def make_connection_vmangos():
    global _db_connection_vmangos
    
    # Если коннекта еще нет, создаем его
    if _db_connection_vmangos is None:
        _db_connection_vmangos = pymysql.connect(
            host=MYSQL_HOST,
            port=int(MYSQL_PORT),
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database='vmangos_db'
        )
    else:
        # Проверяем, не отвалился ли существующий коннект (на всякий случай)
        try:
            _db_connection_vmangos.ping(reconnect=True)
        except Exception:
            # Если отвалился — ping() с флагом reconnect=True поднимет его сам,
            # но если совсем всё плохо, пересоздаем вручную
            _db_connection_vmangos = pymysql.connect(
                host=MYSQL_HOST,
                port=int(MYSQL_PORT),
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                database='vmangos_db'
            )
            
    return _db_connection_vmangos

def make_connection_tbc_cmangos():
    global _db_connection_tbc_cmangos
    
    # Если коннекта еще нет, создаем его
    if _db_connection_tbc_cmangos is None:
        _db_connection_tbc_cmangos = pymysql.connect(
            host=MYSQL_HOST,
            port=int(MYSQL_PORT),
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database='tbcmangos'
        )
    else:
        # Проверяем, не отвалился ли существующий коннект (на всякий случай)
        try:
            _db_connection_tbc_cmangos.ping(reconnect=True)
        except Exception:
            # Если отвалился — ping() с флагом reconnect=True поднимет его сам,
            # но если совсем всё плохо, пересоздаем вручную
            _db_connection_tbc_cmangos = pymysql.connect(
                host=MYSQL_HOST,
                port=int(MYSQL_PORT),
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                database='tbcmangos'
            )
            
    return _db_connection_tbc_cmangos

def is_vanilla_broadcast_text(text_id, text):
    """Check if text is vanilla content with a single query"""
    db = make_connection_vmangos()
    cursor = db.cursor()
    
    query = """
        SELECT 1
        FROM broadcast_text bt
        WHERE bt.entry = %s
        LIMIT 1
    """
    
    cursor.execute(query, (text_id))
    result = cursor.fetchone()

    if result is None:

        import re
        from difflib import SequenceMatcher
        clean_text = re.sub(r'\s+', '', text)

        query = """
            SELECT 1
            FROM quest_greeting qg
            WHERE REPLACE(qg.content_default, ' ', '') = %s
            LIMIT 1
        """
    
        cursor.execute(query, (clean_text))
        result = cursor.fetchone()

    db.close()
    return result is not None

def is_tbc_broadcast_text(text_id, text):
    """Check if text is tbc content with a single query"""
    db = make_connection_tbc_cmangos()
    cursor = db.cursor()
    
    query = """
        SELECT 1
        FROM broadcast_text bt
        WHERE bt.id = %s
        LIMIT 1
    """
    
    cursor.execute(query, (text_id))
    result = cursor.fetchone()

    if result is None:

        import re
        clean_text = re.sub(r'\s+', '', text)

        query = """
            SELECT 1
            FROM questgiver_greeting qg
            WHERE REPLACE(qg.Text, ' ', '') = %s
            LIMIT 1
        """
        cursor.execute(query, (clean_text))
        result = cursor.fetchone()
    
    db.close()
    return result is not None

def is_wotlk_broadcast_text(text_id, text):
    """Check if text is tbc content with a single query"""
    db = make_connection()
    cursor = db.cursor()
    
    query = """
        SELECT 1
        FROM broadcast_text bt
        WHERE bt.id = %s
        LIMIT 1
    """
    
    cursor.execute(query, (text_id))
    result = cursor.fetchone()

    if result is None:

        import re
        clean_text = re.sub(r'\s+', '', text)

        query = """
            SELECT 1
            FROM quest_greeting qg
            WHERE REPLACE(qg.Greeting, ' ', '') = %s
            LIMIT 1
        """
        cursor.execute(query, (clean_text))
        result = cursor.fetchone()
    
    db.close()
    return result is not None

def get_vanilla_quest_text(db_con, quest_id, text_type):
    """Check if text is vanilla content with a single query"""
    cursor = db_con.cursor()

    mapping = {
        'accept': 'Details',
        'complete': 'OfferRewardText',
        'progress': 'RequestItemsText'
    }
    
    query = f"""
        SELECT {mapping.get(text_type.lower())}
        FROM quest_template qt
        WHERE qt.entry = %s
        LIMIT 1
    """
    
    cursor.execute(query, quest_id)
    result = cursor.fetchone()

    return result

def get_tbc_quest_text(db_con, quest_id, text_type):
    """Check if text is vanilla content with a single query"""
    cursor = db_con.cursor()

    mapping = {
        'accept': 'Details',
        'complete': 'OfferRewardText',
        'progress': 'RequestItemsText'
    }
    
    query = f"""
        SELECT {mapping.get(text_type.lower())}
        FROM quest_template qt
        WHERE qt.entry = %s
        LIMIT 1
    """
    
    cursor.execute(query, quest_id)
    result = cursor.fetchone()

    return result

def make_connection_turtle():
    global _db_connection_turtle
    
    # Если коннекта еще нет, создаем его
    if _db_connection_turtle is None:
        _db_connection_turtle = pymysql.connect(
            host=MYSQL_HOST,
            port=int(MYSQL_PORT),
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=TURTLE_DB_NAME
        )
    else:
        # Проверяем, не отвалился ли существующий коннект (на всякий случай)
        try:
            _db_connection_turtle.ping(reconnect=True)
        except Exception:
            # Если отвалился — ping() с флагом reconnect=True поднимет его сам,
            # но если совсем всё плохо, пересоздаем вручную
            _db_connection_turtle = pymysql.connect(
                host=MYSQL_HOST,
                port=int(MYSQL_PORT),
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                database=TURTLE_DB_NAME
            )
            
    return _db_connection_turtle

def get_npc_model_id():
    db = make_connection()
    cursor = db.cursor()
    
    query = """
        SELECT 1
        FROM broadcast_text bt
        WHERE bt.id = %s
        LIMIT 1
    """
    
    cursor.execute(query, (text_id))
    result = cursor.fetchone()

    if result is None:

        import re
        clean_text = re.sub(r'\s+', '', text)

        query = """
            SELECT 1
            FROM quest_greeting qg
            WHERE REPLACE(qg.Greeting, ' ', '') = %s
            LIMIT 1
        """
        cursor.execute(query, (clean_text))
        result = cursor.fetchone()
    
    db.close()
    return result is not None

def query_dataframe_for_all_quests_and_gossip(lang: int = 0):
    db = make_connection()
    sql_query = '''
WITH RECURSIVE
creature_quest_relations AS (
    SELECT 'accept' as source, qr.quest, ct.entry as creature_id
    FROM creature_template ct
    JOIN creature_queststarter qr ON qr.id = ct.entry
        UNION ALL
    SELECT 'accept' as source, qr.quest, ct.entry as creature_id
    FROM creature_template ct
    JOIN game_event_creature_quest qr ON qr.id = ct.entry
        UNION ALL
    SELECT 'complete' as source, qr.quest, ct.entry as creature_id
    FROM creature_template ct
    JOIN creature_questender qr ON qr.id = ct.entry
        UNION ALL
    SELECT 'progress' as source, qr.quest, ct.entry as creature_id
    FROM creature_template ct
    JOIN creature_questender qr ON qr.id = ct.entry
),
gameobject_quest_relations AS (
    SELECT 'accept' as source, qr.quest, gt.entry as gameobject_id
    FROM gameobject_template gt
    JOIN gameobject_queststarter qr ON qr.id = gt.entry
        UNION ALL
    SELECT 'complete' as source, qr.quest, gt.entry as gameobject_id
    FROM gameobject_template gt
    JOIN gameobject_questender qr ON qr.id = gt.entry
        UNION ALL
    SELECT 'progress' as source, qr.quest, gt.entry as gameobject_id
    FROM gameobject_template gt
    JOIN gameobject_questender qr ON qr.id = gt.entry
),
item_quest_relations AS (
    SELECT 'accept' as source, it.startquest as quest, it.entry as item_id
    FROM item_template it
    WHERE it.startquest
),
collected_gossip_menus (base_menu_id, menu_id, text_id, action_menu_id) AS (
    WITH gossip_menu_and_options AS (
        SELECT gm.MenuID, gm.TextID, NULL as action_menu_id
            FROM gossip_menu gm
        UNION DISTINCT
        SELECT gm.MenuID, gm.TextID, gmo.ActionMenuID
            FROM gossip_menu gm
            LEFT JOIN gossip_menu_option gmo on gmo.MenuID = gm.MenuID
            WHERE gmo.OptionType NOT IN (5)
    )
    SELECT gm.MenuID as base_menu_id, gm.MenuID, gm.TextID, gm.action_menu_id
        FROM gossip_menu_and_options gm
    UNION DISTINCT
    SELECT cgm.base_menu_id, gm.MenuID, gm.TextID, gm.action_menu_id
        FROM gossip_menu_and_options gm
        INNER JOIN collected_gossip_menus cgm ON cgm.action_menu_id = gm.MenuID
),
creature_data AS (
    SELECT
        ct.entry as id,
        ct.name,
        cgm.text_id,
        cdie.DisplaySexID,
        cdie.DisplayRaceID,
        ct.Expansion as expansion,
        ct.Zone as zone,
        se.Name as voice_type
        FROM creature_template ct
   	    JOIN creature_template_model ctm ON ct.entry = ctm.CreatureID
        JOIN CreatureDisplayInfo2 cdi ON ctm.CreatureDisplayID = cdi.ID
        LEFT JOIN CreatureDisplayInfoExtra2 cdie ON cdi.ExtendedDisplayInfoID = cdie.ID
        LEFT JOIN collected_gossip_menus cgm ON cgm.base_menu_id = ct.gossip_menu_id
        LEFT JOIN dbc_npcsounds ns ON cdi.NPCSoundID = ns.ID
        LEFT JOIN dbc_soundentries se ON ns.SoundID_1 = se.ID
),
gameobject_data AS (
    SELECT
        gt.entry as id,
        gt.name,
        cgm.text_id,
        gt.Expansion as expansion,
        gt.Zone as zone
    FROM gameobject_template gt
        LEFT JOIN collected_gossip_menus cgm ON cgm.base_menu_id =
            CASE gt.type
                WHEN 2  THEN data3  -- GAMEOBJECT_TYPE_QUESTGIVER (type 2) has property "gossipID" in data3 field
                WHEN 8  THEN data10 -- GAMEOBJECT_TYPE_SPELL_FOCUS (type 8) has property "gossipID" in data10 field
                WHEN 10 THEN data19 -- GAMEOBJECT_TYPE_GOOBER (type 10) has property "gossipID" in data19 field
            END
    WHERE gt.type IN (2, 10)
),
numbers AS (
    SELECT 0 AS n
    UNION ALL SELECT 1
    UNION ALL SELECT 2
    UNION ALL SELECT 3
    UNION ALL SELECT 4
    UNION ALL SELECT 5
    UNION ALL SELECT 6
    UNION ALL SELECT 7
),
ALL_DATA AS (

-- Creature QuestGivers

SELECT
    distinct
    qr.source,
    qr.quest,
    qt.LogTitle as quest_title,
    CASE
        WHEN qr.source = 'accept' THEN qt.QuestDescription
        WHEN qr.source = 'progress' THEN qt.CompletionText
        ELSE qt.RewardText
    END as "text",
    0 as broadcast_text_id,
    cdie.DisplayRaceID,
    cdie.DisplaySexID,
    ct.name,
    'creature' as type,
    qr.creature_id as id,
    qt.Expansion as expansion,
    at.AreaName_Lang_enUS as zone,
    se.Name as voice_type
FROM
    creature_quest_relations qr
JOIN quest_template qt ON qr.quest = qt.ID
JOIN creature_template ct ON qr.creature_id = ct.entry
JOIN creature_template_model ctm ON ct.entry = ctm.CreatureID
JOIN CreatureDisplayInfo2 cdi ON ctm.CreatureDisplayID = cdi.ID
LEFT JOIN wotlkmangos.dbc_areatable at ON qt.QuestSortID = at.ID 
LEFT JOIN CreatureDisplayInfoExtra2 cdie ON cdi.ExtendedDisplayInfoID = cdie.ID
LEFT JOIN dbc_npcsounds ns ON cdi.NPCSoundID = ns.ID
LEFT JOIN dbc_soundentries se ON ns.SoundID_1 = se.ID

WHERE
    (
        (qr.source = 'accept' AND qt.QuestDescription IS NOT NULL AND qt.QuestDescription != '')
        OR (qr.source = 'progress' AND qt.CompletionText IS NOT NULL AND qt.CompletionText != '')
        OR (qr.source = 'complete' AND qt.RewardText IS NOT NULL AND qt.RewardText != '')
    )

-- GameObject QuestGivers

UNION ALL
SELECT
    distinct
    qr.source,
    qr.quest,
    qt.LogTitle as quest_title,
    CASE
        WHEN qr.source = 'accept' THEN qt.QuestDescription
        WHEN qr.source = 'progress' THEN qt.CompletionText
        ELSE qt.RewardText
    END as "text",
    0 as broadcast_text_id,
    -1 as DisplayRaceID,
    0 as DisplaySexID,
    gt.name,
    'gameobject' as type,
    qr.gameobject_id as id,
    qt.Expansion as expansion,
    at.AreaName_Lang_enUS as zone,
    '' as voice_type
FROM
    gameobject_quest_relations qr
JOIN quest_template qt ON qr.quest = qt.ID
JOIN gameobject_template gt ON qr.gameobject_id = gt.entry
LEFT JOIN wotlkmangos.dbc_areatable at ON qt.QuestSortID = at.ID 

WHERE
   (
        (qr.source = 'accept' AND qt.QuestDescription IS NOT NULL AND qt.QuestDescription != '')
       OR (qr.source = 'progress' AND qt.CompletionText IS NOT NULL AND qt.CompletionText != '')
       OR (qr.source = 'complete' AND qt.RewardText IS NOT NULL AND qt.RewardText != '')
  )

-- Item QuestGivers

UNION ALL
SELECT
    distinct
    qr.source,
    qr.quest,
    qt.LogTitle as quest_title,
    qt.QuestDescription as "text",
    0 as broadcast_text_id,
    -1 as DisplayRaceID,
    0 as DisplaySexID,
    it.name,
    'item' as type,
    qr.item_id as id,
    qt.Expansion as expansion,
    at.AreaName_Lang_enUS as zone,
    '' as voice_type
FROM
    item_quest_relations qr
JOIN quest_template qt ON qr.quest = qt.ID
JOIN item_template it ON qr.item_id = it.entry
LEFT JOIN wotlkmangos.dbc_areatable at ON qt.QuestSortID = at.ID 

WHERE
    (
        (qr.source = 'accept' AND qt.QuestDescription IS NOT NULL AND qt.QuestDescription != '')
    )

-- Creature Gossip

UNION ALL
SELECT
    distinct
    'gossip' as source,
    '' as quest,
    '' as quest_title,
    CASE
     WHEN creature_data.DisplaySexID = 0 THEN 
        COALESCE(NULLIF(bt.maletext, ''), NULLIF(bt.femaletext, ''), '')
     WHEN creature_data.DisplaySexID = 1 THEN 
        COALESCE(NULLIF(bt.femaletext, ''), NULLIF(bt.maletext, ''), '')
     ELSE COALESCE(NULLIF(bt.maletext, ''), NULLIF(bt.femaletext, ''), '')
	END AS TEXT,
    bt.ID as broadcast_text_id,
    creature_data.DisplayRaceID,
    creature_data.DisplaySexID,
    creature_data.name,
    'creature' as type,
    creature_data.id,
    creature_data.expansion as expansion,
    at.AreaName_Lang_enUS as zone,
    creature_data.voice_type
FROM creature_data
    CROSS JOIN numbers
    JOIN npc_text nt ON nt.ID = creature_data.text_id
    LEFT JOIN wotlkmangos.dbc_areatable at ON creature_data.Zone = at.ID
    JOIN broadcast_text bt ON
        CASE numbers.n
            WHEN 0 THEN nt.BroadcastTextID0
            WHEN 1 THEN nt.BroadcastTextID1
            WHEN 2 THEN nt.BroadcastTextID2
            WHEN 3 THEN nt.BroadcastTextID3
            WHEN 4 THEN nt.BroadcastTextID4
            WHEN 5 THEN nt.BroadcastTextID5
            WHEN 6 THEN nt.BroadcastTextID6
            WHEN 7 THEN nt.BroadcastTextID7
        END = bt.ID

WHERE
    (DisplaySexID = 0 AND (bt.maletext IS NOT NULL AND bt.maletext != '') OR (bt.femaletext IS NOT NULL AND bt.femaletext != ''))
    OR (DisplaySexID = 1 AND (bt.femaletext IS NOT NULL AND bt.femaletext != '') OR (bt.maletext IS NOT NULL AND bt.maletext != ''))
    OR (DisplaySexID IS NULL AND (bt.maletext IS NOT NULL AND bt.maletext != '') OR (bt.femaletext IS NOT NULL AND bt.femaletext != ''))


-- GameObject Gossip

UNION ALL
SELECT
    distinct
    'gossip' as source,
    '' as quest,
    '' as quest_title,
    IF(bt.MaleText IS NOT NULL AND bt.MaleText != '', bt.MaleText, bt.FemaleText) AS text,
    bt.ID as broadcast_text_id,
    -1 as DisplayRaceID,
    0 as DisplaySexID,
    gameobject_data.name,
    'gameobject' as type,
    gameobject_data.id,
    gameobject_data.Expansion as expansion,
    at.AreaName_Lang_enUS as zone,
    '' as voice_type
FROM gameobject_data
    CROSS JOIN numbers
    JOIN npc_text nt ON nt.ID = gameobject_data.text_id
    LEFT JOIN wotlkmangos.dbc_areatable at ON gameobject_data.Zone = at.ID
    JOIN broadcast_text bt ON
        CASE numbers.n
            WHEN 0 THEN nt.BroadcastTextID0
            WHEN 1 THEN nt.BroadcastTextID1
            WHEN 2 THEN nt.BroadcastTextID2
            WHEN 3 THEN nt.BroadcastTextID3
            WHEN 4 THEN nt.BroadcastTextID4
            WHEN 5 THEN nt.BroadcastTextID5
            WHEN 6 THEN nt.BroadcastTextID6
            WHEN 7 THEN nt.BroadcastTextID7
        END = bt.ID

WHERE
    bt.MaleText IS NOT NULL AND bt.MaleText != '' OR
    bt.FemaleText IS NOT NULL AND bt.FemaleText != ''

-- Creature QuestGreetings

UNION ALL
SELECT
    distinct
    'gossip' as source,
    '' as quest,
    '' as quest_title,
    qg.Greeting AS text,
    0 as broadcast_text_id,
    creature_data.DisplayRaceID,
    creature_data.DisplaySexID,
    creature_data.name,
    'creature' as type,
    creature_data.id,
    creature_data.Expansion as expansion,
    at.AreaName_Lang_enUS as zone,
    creature_data.voice_type
FROM creature_data
    JOIN quest_greeting qg ON qg.ID=creature_data.id AND type=0
    LEFT JOIN wotlkmangos.dbc_areatable at ON creature_data.Zone = at.ID

-- GameObject QuestGreetings

UNION ALL
SELECT
    distinct
    'gossip' as source,
    '' as quest,
    '' as quest_title,
    qg.Greeting AS text,
    0 as broadcast_text_id,
    -1 AS DisplayRaceID,
    0 AS DisplaySexID,
    gameobject_data.name,
    'gameobject' as type,
    gameobject_data.id,
    gameobject_data.Expansion as expansion,
    at.AreaName_Lang_enUS as zone,
    '' as voice_type
FROM gameobject_data
    JOIN quest_greeting qg ON qg.ID=gameobject_data.id AND type=1
    LEFT JOIN wotlkmangos.dbc_areatable at ON gameobject_data.Zone = at.ID

)
    '''

    if lang == 0:
        sql_query += '''
SELECT
    source,
    quest,
    quest_title,
    text,
    broadcast_text_id,
    DisplayRaceID,
    DisplaySexID,
    name,
    type,
    id,
    text as original_text,
    expansion,
    zone,
    voice_type
FROM ALL_DATA
        '''
    else:
        sql_query += f'''
SELECT
    source,
    quest,
    IFNULL(NULLIF(lq.title_loc{lang}, ''), quest_title) as quest_title,
    IFNULL(NULLIF(CASE source
        WHEN 'gossip' THEN (CASE
            WHEN broadcast_text_id = 0 THEN qg.content_loc{lang}
            WHEN ALL_DATA.type = 'creature' THEN IF(DisplaySexID = 0, lbt.male_text_loc{lang}, lbt.female_text_loc{lang})
            ELSE IFNULL(NULLIF(lbt.male_text_loc{lang}, ''), lbt.female_text_loc{lang})
        END)
        WHEN 'accept'   THEN lq.Details_loc{lang}
        WHEN 'progress' THEN lq.RequestItemsText_loc{lang}
        WHEN 'complete' THEN lq.OfferRewardText_loc{lang}
        ELSE NULL
    END, ''), text) as text,
    DisplayRaceID,
    DisplaySexID,
    IFNULL(NULLIF(CASE ALL_DATA.type
        WHEN 'creature'   THEN lc.name_loc{lang}
        WHEN 'gameobject' THEN lg.name_loc{lang}
        WHEN 'item'       THEN li.name_loc{lang}
        ELSE NULL
    END, ''), name) as name,
    ALL_DATA.type,
    id,
    text as original_text
FROM ALL_DATA
    LEFT JOIN wotlk_mangos.locales_quest          lq  ON lq .entry = quest
    LEFT JOIN wotlk_mangos.locales_broadcast_text lbt ON lbt.entry = broadcast_text_id
    LEFT JOIN wotlk_mangos.locales_creature       lc  ON lc .entry = id AND type = 'creature'
    LEFT JOIN wotlk_mangos.locales_gameobject     lg  ON lg .entry = id AND type = 'gameobject'
    LEFT JOIN wotlk_mangos.locales_item           li  ON li .entry = id AND type = 'item'
    LEFT JOIN wotlk_mangos.quest_greeting         qg  ON qg .entry = id AND qg.type = (CASE ALL_DATA.type WHEN 'creature' THEN 0 WHEN 'gameobject' THEN 1 ELSE -1 END)
        '''

    with db.cursor() as cursor:
        cursor.execute(sql_query)
        data = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

    db.close()
    df = pd.DataFrame(data, columns=columns)

    return df

def query_dataframe_for_turtle(lang: int = 0):
    db = make_connection_turtle()
    sql_query = f'''
WITH RECURSIVE
creature_quest_relations AS (
    SELECT 'accept' as source, qr.quest, ct.entry as creature_id
    FROM creature_template ct
    JOIN creature_questrelation qr ON qr.id = ct.entry
        UNION ALL
    -- SELECT 'accept' as source, qr.quest, ct.entry as creature_id
    -- FROM creature_template ct
    -- JOIN game_event_creature_quest qr ON qr.id = ct.entry
    --     UNION ALL
    SELECT 'complete' as source, qr.quest, ct.entry as creature_id
    FROM creature_template ct
    JOIN creature_involvedrelation qr ON qr.id = ct.entry
        UNION ALL
    SELECT 'progress' as source, qr.quest, ct.entry as creature_id
    FROM creature_template ct
    JOIN creature_involvedrelation qr ON qr.id = ct.entry
),
gameobject_quest_relations AS (
    SELECT 'accept' as source, qr.quest, gt.entry as gameobject_id
    FROM gameobject_template gt
    JOIN gameobject_questrelation qr ON qr.id = gt.entry
        UNION ALL
    SELECT 'complete' as source, qr.quest, gt.entry as gameobject_id
    FROM gameobject_template gt
    JOIN gameobject_involvedrelation qr ON qr.id = gt.entry
        UNION ALL
    SELECT 'progress' as source, qr.quest, gt.entry as gameobject_id
    FROM gameobject_template gt
    JOIN gameobject_involvedrelation qr ON qr.id = gt.entry
),
item_quest_relations AS (
    SELECT 'accept' as source, it.start_quest as quest, it.entry as item_id
    FROM item_template it
    WHERE it.start_quest
),
collected_gossip_menus (base_menu_id, menu_id, text_id, action_menu_id) AS (
    WITH gossip_menu_and_options AS (
        SELECT gm.entry as MenuID, gm.text_id as TextID, NULL as action_menu_id
            FROM gossip_menu gm
        UNION DISTINCT
        SELECT gm.entry as MenuID, gm.text_id as TextID, gmo.action_menu_id as ActionMenuID
            FROM gossip_menu gm
            LEFT JOIN gossip_menu_option gmo on gmo.menu_id = gm.entry
            WHERE gmo.option_id NOT IN (5)
    )
    SELECT gm.MenuID as base_menu_id, gm.MenuID, gm.TextID, gm.action_menu_id
        FROM gossip_menu_and_options gm
    UNION DISTINCT
    SELECT cgm.base_menu_id, gm.MenuID, gm.TextID, gm.action_menu_id
        FROM gossip_menu_and_options gm
        INNER JOIN collected_gossip_menus cgm ON cgm.action_menu_id = gm.MenuID
),
creature_data AS (
    SELECT
        ct.entry as id,
        ct.name,
        cgm.text_id,
        cdie.DisplaySexID,
        cdie.DisplayRaceID,
        ct.expansion as expansion,
        ct.Zone as zone,
        se.Name as voice_type
        FROM creature_template ct
   	    -- JOIN creature_template_model ctm ON ct.entry = ctm.CreatureID
        JOIN CreatureDisplayInfo cdi ON ct.display_id1 = cdi.ID
        LEFT JOIN CreatureDisplayInfoExtra cdie ON cdi.ExtendedDisplayInfoID = cdie.ID
        LEFT JOIN collected_gossip_menus cgm ON cgm.base_menu_id = ct.gossip_menu_id
        LEFT JOIN dbc_npcsounds ns ON cdi.NPCSoundID = ns.ID
        LEFT JOIN dbc_soundentries se ON ns.SoundID_1 = se.ID
),
gameobject_data AS (
    SELECT
        gt.entry as id,
        gt.name,
        cgm.text_id,
        gt.expansion as expansion,
        gt.Zone as zone
    FROM gameobject_template gt
        LEFT JOIN collected_gossip_menus cgm ON cgm.base_menu_id =
            CASE gt.type
                WHEN 2  THEN data3  -- GAMEOBJECT_TYPE_QUESTGIVER (type 2) has property "gossipID" in data3 field
                WHEN 8  THEN data10 -- GAMEOBJECT_TYPE_SPELL_FOCUS (type 8) has property "gossipID" in data10 field
                WHEN 10 THEN data19 -- GAMEOBJECT_TYPE_GOOBER (type 10) has property "gossipID" in data19 field
            END
    WHERE gt.type IN (2, 10)
),
numbers AS (
    SELECT 0 AS n
    UNION ALL SELECT 1
    UNION ALL SELECT 2
    UNION ALL SELECT 3
    UNION ALL SELECT 4
    UNION ALL SELECT 5
    UNION ALL SELECT 6
    UNION ALL SELECT 7
),
ALL_DATA AS (

-- Creature QuestGivers

SELECT
    distinct
    qr.source,
    qr.quest,
    qt.Title as quest_title,
    CASE
        WHEN qr.source = 'accept' THEN qt.Details
        WHEN qr.source = 'progress' THEN qt.RequestItemsText
        ELSE qt.OfferRewardText
    END as "text",
    0 as broadcast_text_id,
    cdie.DisplayRaceID,
    cdie.DisplaySexID,
    ct.name,
    'creature' as type,
    qr.creature_id as id,
    qt.Expansion as expansion,
    at.AreaName_enUS as zone,
    se.Name as voice_type
FROM
    creature_quest_relations qr
JOIN quest_template qt ON qr.quest = qt.entry
JOIN creature_template ct ON qr.creature_id = ct.entry
-- JOIN creature_template_model ctm ON ct.entry = ctm.CreatureID
JOIN CreatureDisplayInfo cdi ON ct.display_id1 = cdi.ID
LEFT JOIN {TURTLE_DB_NAME}.dbc_areatable at ON qt.ZoneOrSort = at.ID 
LEFT JOIN CreatureDisplayInfoExtra cdie ON cdi.ExtendedDisplayInfoID = cdie.ID
LEFT JOIN dbc_npcsounds ns ON cdi.NPCSoundID = ns.ID
LEFT JOIN dbc_soundentries se ON ns.SoundID_1 = se.ID

WHERE
    (
        (qr.source = 'accept' AND qt.Details IS NOT NULL AND qt.Details != '')
        OR (qr.source = 'progress' AND qt.RequestItemsText IS NOT NULL AND qt.RequestItemsText != '')
        OR (qr.source = 'complete' AND qt.OfferRewardText IS NOT NULL AND qt.OfferRewardText != '')
    )

-- GameObject QuestGivers

UNION ALL
SELECT
    distinct
    qr.source,
    qr.quest,
    qt.Title as quest_title,
    CASE
        WHEN qr.source = 'accept' THEN qt.Details
        WHEN qr.source = 'progress' THEN qt.RequestItemsText
        ELSE qt.OfferRewardText
    END as "text",
    0 as broadcast_text_id,
    -1 as DisplayRaceID,
    0 as DisplaySexID,
    gt.name,
    'gameobject' as type,
    qr.gameobject_id as id,
    qt.Expansion as expansion,
    at.AreaName_enUS as zone,
    '' as voice_type
FROM
    gameobject_quest_relations qr
JOIN quest_template qt ON qr.quest = qt.entry
JOIN gameobject_template gt ON qr.gameobject_id = gt.entry
LEFT JOIN {TURTLE_DB_NAME}.dbc_areatable at ON qt.ZoneOrSort = at.ID 

WHERE
   (
        (qr.source = 'accept' AND qt.Details IS NOT NULL AND qt.Details != '')
       OR (qr.source = 'progress' AND qt.RequestItemsText IS NOT NULL AND qt.RequestItemsText != '')
       OR (qr.source = 'complete' AND qt.OfferRewardText IS NOT NULL AND qt.OfferRewardText != '')
  )

-- Item QuestGivers

UNION ALL
SELECT
    distinct
    qr.source,
    qr.quest,
    qt.Title as quest_title,
    qt.Details as "text",
    0 as broadcast_text_id,
    -1 as DisplayRaceID,
    0 as DisplaySexID,
    it.name,
    'item' as type,
    qr.item_id as id,
    qt.Expansion as expansion,
    at.AreaName_enUS as zone,
    '' as voice_type
FROM
    item_quest_relations qr
JOIN quest_template qt ON qr.quest = qt.entry
JOIN item_template it ON qr.item_id = it.entry
LEFT JOIN {TURTLE_DB_NAME}.dbc_areatable at ON qt.ZoneOrSort = at.ID 

WHERE
    (
        (qr.source = 'accept' AND qt.Details IS NOT NULL AND qt.Details != '')
    )

-- Creature Gossip

UNION ALL
SELECT
    distinct
    'gossip' as source,
    '' as quest,
    '' as quest_title,
    CASE
     WHEN creature_data.DisplaySexID = 0 THEN 
        COALESCE(NULLIF(bt.male_text, ''), NULLIF(bt.female_text, ''), '')
     WHEN creature_data.DisplaySexID = 1 THEN 
        COALESCE(NULLIF(bt.female_text, ''), NULLIF(bt.male_text, ''), '')
     ELSE COALESCE(NULLIF(bt.male_text, ''), NULLIF(bt.female_text, ''), '')
	END AS TEXT,
    bt.entry as broadcast_text_id,
    creature_data.DisplayRaceID,
    creature_data.DisplaySexID,
    creature_data.name,
    'creature' as type,
    creature_data.id,
    creature_data.expansion as expansion,
    at.AreaName_enUS as zone,
    creature_data.voice_type
FROM creature_data
    CROSS JOIN numbers
    JOIN npc_text nt ON nt.ID = creature_data.text_id
    LEFT JOIN {TURTLE_DB_NAME}.dbc_areatable at ON creature_data.Zone = at.ID
    JOIN broadcast_text bt ON
        CASE numbers.n
            WHEN 0 THEN nt.BroadcastTextID0
            WHEN 1 THEN nt.BroadcastTextID1
            WHEN 2 THEN nt.BroadcastTextID2
            WHEN 3 THEN nt.BroadcastTextID3
            WHEN 4 THEN nt.BroadcastTextID4
            WHEN 5 THEN nt.BroadcastTextID5
            WHEN 6 THEN nt.BroadcastTextID6
            WHEN 7 THEN nt.BroadcastTextID7
        END = bt.entry

WHERE
    (DisplaySexID = 0 AND (bt.male_text IS NOT NULL AND bt.male_text != '') OR (bt.female_text IS NOT NULL AND bt.female_text != ''))
    OR (DisplaySexID = 1 AND (bt.female_text IS NOT NULL AND bt.female_text != '') OR (bt.male_text IS NOT NULL AND bt.male_text != ''))
    OR (DisplaySexID IS NULL AND (bt.male_text IS NOT NULL AND bt.male_text != '') OR (bt.female_text IS NOT NULL AND bt.female_text != ''))


-- GameObject Gossip

UNION ALL
SELECT
    distinct
    'gossip' as source,
    '' as quest,
    '' as quest_title,
    IF(bt.male_text IS NOT NULL AND bt.male_text != '', bt.male_text, bt.female_text) AS text,
    bt.entry as broadcast_text_id,
    -1 as DisplayRaceID,
    0 as DisplaySexID,
    gameobject_data.name,
    'gameobject' as type,
    gameobject_data.id,
    gameobject_data.Expansion as expansion,
    at.AreaName_enUS as zone,
    '' as voice_type
FROM gameobject_data
    CROSS JOIN numbers
    JOIN npc_text nt ON nt.ID = gameobject_data.text_id
    LEFT JOIN {TURTLE_DB_NAME}.dbc_areatable at ON gameobject_data.Zone = at.ID
    JOIN broadcast_text bt ON
        CASE numbers.n
            WHEN 0 THEN nt.BroadcastTextID0
            WHEN 1 THEN nt.BroadcastTextID1
            WHEN 2 THEN nt.BroadcastTextID2
            WHEN 3 THEN nt.BroadcastTextID3
            WHEN 4 THEN nt.BroadcastTextID4
            WHEN 5 THEN nt.BroadcastTextID5
            WHEN 6 THEN nt.BroadcastTextID6
            WHEN 7 THEN nt.BroadcastTextID7
        END = bt.entry

WHERE
    bt.male_text IS NOT NULL AND bt.male_text != '' OR
    bt.female_text IS NOT NULL AND bt.female_text != ''

-- Creature QuestGreetings

UNION ALL
SELECT
    distinct
    'gossip' as source,
    '' as quest,
    '' as quest_title,
    qg.content_default AS text,
    0 as broadcast_text_id,
    creature_data.DisplayRaceID,
    creature_data.DisplaySexID,
    creature_data.name,
    'creature' as type,
    creature_data.id,
    creature_data.Expansion as expansion,
    at.AreaName_enUS as zone,
    creature_data.voice_type
FROM creature_data
    JOIN quest_greeting qg ON qg.entry=creature_data.id AND type=0
    LEFT JOIN {TURTLE_DB_NAME}.dbc_areatable at ON creature_data.Zone = at.ID

-- GameObject QuestGreetings

UNION ALL
SELECT
    distinct
    'gossip' as source,
    '' as quest,
    '' as quest_title,
    qg.content_default AS text,
    0 as broadcast_text_id,
    -1 AS DisplayRaceID,
    0 AS DisplaySexID,
    gameobject_data.name,
    'gameobject' as type,
    gameobject_data.id,
    gameobject_data.Expansion as expansion,
    at.AreaName_enUS as zone,
    '' as voice_type
FROM gameobject_data
    JOIN quest_greeting qg ON qg.entry=gameobject_data.id AND type=1
    LEFT JOIN {TURTLE_DB_NAME}.dbc_areatable at ON gameobject_data.Zone = at.ID

)
    '''

    if lang == 0:
        sql_query += '''
SELECT
    source,
    quest,
    quest_title,
    text,
    broadcast_text_id,
    DisplayRaceID,
    DisplaySexID,
    name,
    type,
    id,
    text as original_text,
    expansion,
    zone,
    voice_type
FROM ALL_DATA
        '''
    else:
        sql_query += f'''
SELECT
    source,
    quest,
    IFNULL(NULLIF(lq.title_loc{lang}, ''), quest_title) as quest_title,
    IFNULL(NULLIF(CASE source
        WHEN 'gossip' THEN (CASE
            WHEN broadcast_text_id = 0 THEN qg.content_loc{lang}
            WHEN ALL_DATA.type = 'creature' THEN IF(DisplaySexID = 0, lbt.male_text_loc{lang}, lbt.female_text_loc{lang})
            ELSE IFNULL(NULLIF(lbt.male_text_loc{lang}, ''), lbt.female_text_loc{lang})
        END)
        WHEN 'accept'   THEN lq.Details_loc{lang}
        WHEN 'progress' THEN lq.RequestItemsText_loc{lang}
        WHEN 'complete' THEN lq.OfferRewardText_loc{lang}
        ELSE NULL
    END, ''), text) as text,
    DisplayRaceID,
    DisplaySexID,
    IFNULL(NULLIF(CASE ALL_DATA.type
        WHEN 'creature'   THEN lc.name_loc{lang}
        WHEN 'gameobject' THEN lg.name_loc{lang}
        WHEN 'item'       THEN li.name_loc{lang}
        ELSE NULL
    END, ''), name) as name,
    ALL_DATA.type,
    id,
    text as original_text
FROM ALL_DATA
    LEFT JOIN wotlk_mangos.locales_quest          lq  ON lq .entry = quest
    LEFT JOIN wotlk_mangos.locales_broadcast_text lbt ON lbt.entry = broadcast_text_id
    LEFT JOIN wotlk_mangos.locales_creature       lc  ON lc .entry = id AND type = 'creature'
    LEFT JOIN wotlk_mangos.locales_gameobject     lg  ON lg .entry = id AND type = 'gameobject'
    LEFT JOIN wotlk_mangos.locales_item           li  ON li .entry = id AND type = 'item'
    LEFT JOIN wotlk_mangos.quest_greeting         qg  ON qg .entry = id AND qg.type = (CASE ALL_DATA.type WHEN 'creature' THEN 0 WHEN 'gameobject' THEN 1 ELSE -1 END)
        '''

    with db.cursor() as cursor:
        cursor.execute(sql_query)
        data = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

    db.close()
    df = pd.DataFrame(data, columns=columns)

    return df
