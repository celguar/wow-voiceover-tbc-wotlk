# https://wowpedia.fandom.com/wiki/RaceId
RACE_DICT = {
    -1: 'narrator',
    1: 'human',
    2: 'orc',
    3: 'dwarf',
    4: 'nightelf',
    5: 'scourge',
    6: 'tauren',
    7: 'gnome',
    8: 'troll',
    9: 'goblin',
    10: 'bloodelf',
    11: 'draenei',
    12: 'felorc',
    13: 'naga',
    14: 'broken',
    15: 'skeleton',
    16: 'vrykul',
    17: 'tuskarr',
    18: 'foresttroll',
    19: 'taunka',
    20: 'northrendskeleton',
    21: 'icetroll',
    22: 'worgen',
    23: 'human',
    24: 'pandaren',
    25: 'pandaren',
    26: 'pandaren',
    27: 'nightborne',
    28: 'highmountaintauren',
    29: 'voidelf',
    30: 'lightforgeddraenei',
    31: 'zandalari',
    32: 'kultiran',
    33: 'thinhuman',
    34: 'darkirondwarf',
    35: 'vulpera',
    36: 'magharorc',
    37: 'mechagnome',
    38: 'ethereal', #added not in website
    39: 'giant', #added not in website
    40: 'demon', #added not in website
    41: 'nerubian', #added not in website
    42: 'arakkoa', #added not in website
    43: 'furbolg', #added not in website
    44: 'wolvar', #added not in website
    45: 'gorloc', #added not in website
    52: 'dracthyr',
    70: 'dracthyr',
    -77:'custom'
}

GENDER_DICT = {0: 'male',
               1: 'female',
               -77: 'custom'}

RACE_DICT_INV = {v: k for k, v in RACE_DICT.items()}
GENDER_DICT_INV = {v: k for k, v in GENDER_DICT.items()}

def race_gender_tuple_to_strings(race_gender_tuple):
    race_gender_strings = []

    for race_id, gender_id in race_gender_tuple:
        race = RACE_DICT.get(race_id, 'unknown')
        gender = GENDER_DICT.get(gender_id, 'unknown')
        race_gender_strings.append(f"{race}_{gender}")

    return race_gender_strings

REPLACE_DICT = {
                '$b': '\n', '$B': '\n', '$n': 'adventurer', '$N': 'Adventurer',
                '$C': 'Adventurer', '$c': 'adventurer', '$R': 'Traveler', '$r': 'traveler', '$t citizen : citizen': 'citizen',
                '$T Civvy : Civvy;': '',
                '<name>': 'adventurer', '<Name>': 'Adventurer',
                '<race>': 'traveler', '<Race>': 'Traveler',
                '<class>': 'adventurer', '<Class>': 'Adventurer',
                 '—':',', '--':',', " - ":", ",
                 # Factions / Regions
                 "Draenei": "Dray-nai",
                "Lordaeron": "Lor-deron",
                "Quel'Thalas": "Kwel-tha-las",
                "Dalaran": "Dalah-ran",
                "Naxxramas": "Nax-ramas",
                "Scholomance": "Skolo-mance",
                "Stratholme": "Strath-holm",
                "Atal'ai":"Ata-lai",
                "Naaru":"Naroo",
                "Dragonflight": "Dragon-flight",
                "Necrolord":"necro-lord",
                "bloodmage":"blood-mage",
                "taunka'le":"taunka-lay",
                "wyrm":"werm",
                "Oneqwah":"Ohnay-kwah",
                "vrykul":"vrye-kool",
                "thor modan":"thormoe-dhaan",
                "ursoc":"ursok",
                "dun argol":"duunar-goll",
                "earthen":"ehrrthin",

                # Bosses / NPCs
                "Malygos": "Maali-goss",
                "Kel'Thuzad": "Kel-thu-zahd",
                "Anub'arak": "Anoobah-raak",
                "Kael'thas": "Kale-thoss",
                "Mok'Nathal":"Mockna-tholl",
                "orcish":"orkish",
                "Kil'jaeden": "Kil-jayden",
                "Archimonde": "Arki-mond",
                "C'Thun": "Kuh-thoon",
                "Yogg-Saron": "Yog-suh-ron",
                "Gjalerbron": "Yal-er-bron",
                "Heb'Drakkar": "Heb-drah-kar",
                "Rageclaw": "Rage-claw",
                "Ragemane": "Rage-mane",
                "Verna":"Vur-nah",
                "Pathaleon":"Pathalion",
                "Demetrian":"Deh-mee-tree-ahn",
                "Zul'Marosh": "Zool-marosh",
                "Medivh":"Medaeve",
                "Dar'Khan":"DarKahn",
                "Stormrage": "Storm-rayge",
                "Gul'dan":"Gool dan",
                "undead":"on-ded",
                "undeath":"on-deth",
                "Lok'tar ogar":"Loktaro garr",
                'mrgl-mrgl':"mergle-mergle",

                # Places
                #"Azeroth":"Ah-ze-roth",
                "Icecrown": "Ice crown",
                "Dragonshrine": "Dragon-shrine",
                "Auchindoun": "Aw-kin-doon",
                "Hyjal": "High-jahl",
                "Mathystra": "Math-is-trah",
                "Ulduar": "Ool-dwar",
                "Utgarde": "Oot-guard",
                "Zul'Aman": "Zool-ahmaan",
                "Zul'Drak": "Zool-drak",
                "Ahn'kahet": "On-ka-het",
                "Gundrak": "Gun-drak",
                "Modan":"Moe-dahn",
                "Ahn'Qiraj":"On-kee-rahj",
                "Elwynn":"Elwin",
                "Arcatraz":"Arc-a-traz",
                "Stonetalon":"stone-talon",
                "Kalimdor":"Kalim-dor",

                # Titans / Lore
                "Tyr": "Teer",
                "Freya": "Fray-ah",
                "Thorim": "Thor-rim",
                "Hodir": "Ho-deer",
                #"Loken": "Low-ken",
                "Ymiron": "Yee-miron",
                "Elune":"Ehloon",

                # Misc
                "Felwood": "Fell-wood",
                "Ashenvale": "Ashen-veil",
                "Sha'naar":"Shanar",
                "Sin'dorei":"Sindoh-rye",
                "Gorefiend":"Gorfeend",
                "Indu'le": "Indu-lay",


}

REUSE_AUDIO_MAP = {
    "fire_elemental": "demon_male",
    "water_elemental": "demon_male",
    "earth_elemental": "demon_male",
    "wind_elemental": "demon_male",
    "wolf": "rexxar",
    "banshee": "forsaken_female",
    "mountain_giant": "ancient",
    "orc_hero": "felorc_male",
}
VOICE_MODEL_MAP = {
    # sholazar
    "wolvar_male": "rexxar",
    "gorloc_male": "furbolg_male",

    #dragon
    "dragon_female":"tauren_female",
    "dragon_male":"demon_male",


    # big creature shared model
    "tooga":"big_creature",

    "giant_male": "big_creature",
    "vrykul_male": "orc_male",
    "ogre_male": "felorc_male",
    "abomination":"felorc_male",
    "mountain_giant": "felorc_male",
    "mountain_giant_dk": "felorc_male", #for rune giants, e.g., Gavrock
    #other
    "giant_female":"forsaken_female",
    "titan_male":"varian",
    "medivh":"khadgar",
    "ogrila_ogre":'khadgar',
    "earthen":"dwarf_male",
    "naaru":"tauren_female",
    "murloc":"demon_female",
    "fire_elemental":"demon_male",
    "water_elemental":"demon_male",
    "earth_elemental":"demon_male",
    "wind_elemental":"demon_male",
    "wolf":"rexxar",
    "bear":"rexxar",
    "cat":"tauren_female",
    "rhino":"tauren_male",
    "serpent":"naga_female",
    "banshee":"forsaken_female",
}


# maps questgiver IDs to effect types
NPC_EFFECTS = {
    302: "ghost",
    392: "ghost",
    2076:"bubbles",
    2227:"ghost",
    2278:"ghost",
    4606:"ghost",
    6491:"ghost",
    5397:"giant",
    9598:"ghost",
    10666:"undead",
    10684:"ghost",
    10926:"ghost",
    1733:"demon",
    18261:"demon",
    12238:"ghost",
    13716:"ghost",
    14470:"demon",
    14902:"giant",
    16015:"demon",
    14354:"demon",
    16201: "ghost",
    16388: "ghost",
    16813: "ghost",
    16814: "ghost",
    16815: "ghost",
    17712: "ghost",
    17674:"ghost",
    17877:"ancient",
    187565:"ancient",
    18369:"small",
    18445:"small",
    18687:"ghost",
    20812:"small",
    21700:"demon",
    21797:"demon",
    21318:"ghost",
    21330:"demon",
    22103:"demon",
    23778:"undead",
    24137:"undead",
    25425:"ghost",
    26117:"demon",
    26206:"demon",
    24910:"ghost",
    26501:"ghost",
    26471:"ghost",
    27337:"ghost",
    29047:"ghost",
    31135:"ghost",
    19456:"ghost",
    19644:"ghost",
    19488:"undead",
    19489:"undead",
    20463:"undead",
    20464:"undead",
    20482:"comms",
    20518:"comms",
    20084:"comms",
    28377:"undead",
    28943:"undead",
    28907:"undead",
    28911:"undead",
    19937:"ghost",
    29259:"ghost",
    24027:"undead",
    24956:"ghost",
    24261:"wolf", #ulfang
    27275:"bear", #kodian
    27274:"bear", #orsonn
    27350:"demon",
    26443:"demon",
    27950:"demon",
    26917:"demon",
    27575:"demon",
    27990:"demon",
    27785:"demon",
    28666:"undead", #gorebag
    28589:"demon", #gristlegut
    26527:"demon", #chromie (purging of stratholme)
    27856:"demon", #chromie (wyrmrest)
    27765:"demon", #nalice
    26593:"demon", #serinar
    26983:"demon", #aurastrasza
    27763:"demon", #vargastrasz
    28760:"undead", #hargus the gimp
    26653:"undead", #kilix the unraveler
    188419:"ghost", #elder mana'loa
    26500:"ghost", #image of drakuru
    26543:"ghost", #image of drakuru
    26701:"ghost", #image of drakuru
    26787:"ghost", #image of drakuru
    26924:"ghost", #gan'jo
    37779:"undead", #dark ranger loralen
    37780:"undead", #dark ranger vorel

}
