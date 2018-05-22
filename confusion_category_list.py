super_category_list = \
    {'chicken': ['gabao__rice_stir-fried_chicken_and_basil', 'salad_chicken',
                 'spicy_chicken', 'tandoori_chicken',
                 'cheese_katsu', 'chicken_cutlet', 'cream_stew_with_chicken',
                 'chicken_mince_cutlet', 'chicken_nanban',
                 'butter_chicken_curry__lou_only', 'roast_chicken',
                 'fried_chicken', 'of_chicken_meatball__sauce',
                 'tatsuta_fried_chicken', 'teriyaki_chicken',
                 'yakitori_with_sauce', 'yakitori_with_salt',
                 'steamed_chicken_breast', 'chicken_steamed', 'youlin_tori'],
     'chilled_noodles': ['forced_udon', 'topped_udon'],
     }


"""
done:
'instant_food':['cup_noodles', 'cup_chow_mein', 'soup_vermicelli'],
'jelly-pudding':['acai_bowl', 'coffee_jelly', 'pudding', 'fruit_jelly', 'almond_junket', 'baked_pudding'],
'chocolate':['chocolate', 'white_chocolate'],
'curry_rice':['chicken_curry', 'beef_curry', 'pork_curry'],
'egg_dish':['omelette', 'scrambled_eggs', 'eyeball_grilled', 'boiled_egg', 'hot_spring_egg', 'thickness_baked_eggs', 'out_maki_tamago',
                 'raw_egg', 'fried_eggs', 'cheeseomelette'],
'fastfood':['chicken_nugget', 'teriyakibaga', 'french_fries'],
'fresh_fruits':['strawberry', 'kiwi', 'banana', 'mandarin_orange', 'apple'],
'fried_food':['kakiage', 'croquette', 'hash_browns', 'beef_croquette', 'spring_roll', 'candied_sweet_potato', 'deep-fried_horse_mackerel', 'fried_shrimp', 'deep_fried_oysters', 'tempura_assorted'],
'gratin-doria':['shrimp_gratin', 'doria', 'potato_gratin', 'lasagna', 'vegetable_gratin'],
'grilled_food-grilled_fish':['potatoes_butter', 'baked_pumpkin', 'grilled_mackerel', 'horse_mackerel_of_dried_fish', 'grilled_salmon', 'grilled_yellowtail', 'yellowtail_teriyaki', 'opening_of_hockey'],

'drinks':['cafe_au_lait', 'caffe_latte', 'black_coffee', 'black_tea', 'green_tea', 'milk', 'yogurt', 'coffee_milk'],
'manufactured_food':['vienna_sausage', 'sausage_saute', 'ham', 'frankfurt_sausage', 'bacon_saute', 'raw_ham_and_prosciutto'],
'croquette':['cream_croquette', 'croquette', 'beef_croquette'],
'minced_meat':['cream_croquette', 'shumai', 'hamburger_meat', 'ground_meat_cutlet', 'grilled_dumplings', 'meat_balls', 'mabo_vermicelli'],
'miso_soup':['miso_soup_with_pork_and_vegetables', 'miso_soup'],
'nut':['almond', 'walnut', 'mixed_nuts'],
'ochazuke-hodgepodge':['ochazuke', 'egg_porridge', 'hodgepodge'],
'Okonomiyaki':['monjayaki', 'okonomiyaki'],
'pilaf-fried_rice':['pilaf_of_clams', 'omelette_rice', 'fried_rice', 'paella'],
'pizza':['pizza__seafood', 'pizza__margherita', 'pizza_toast', 'mix_pizza'],
'pork':['pork_cutlet', 'twice_cooked_pork', 'pork_saute', 'roasted_pork', 'pig_shabu', 'boiled_pork', 'ginger_grilled_pork', 'fried_meat_leek', 'miso_katsu'],
'ramen':['tonkotsu_ramen', 'pork_soy_sauce_ramen', 'salt_ramen', 'soy_source_ramen', 'miso_ramen'],
'rice_ball':['rice_balls__salmon', 'rice_balls__plum', 'rice', 'hand-rolled_sushi'],
'sandwich':['ham_and_vegetables_sand', 'hamburger', 'mix_sand'],
'sesame_vegetbles':['green_beans_sesame', 'spinach_sesame', 'sesame_of_japanese_mustard_spinach'],
'snack-rice_cracker':['popcorn', 'crisps'],
'soba':['kakiagesoba', 'kakesoba', 'zarusoba', 'tanukisoba', 'tororo_buckwheat'],
'stir_fry':['goya_chample', 'stir-fried_bean_sprouts', 'fried_rebanira', 'fried_pig_kimchi', 'stir-fried_bean_sprouts_of_pork',
                 'fried_meat_vegetables__pork', 'mapo_eggplant', 'fried_vegetables', 'fried_chili_sauce_of_shrimp', 'eight_dishes'],
'sweet_bread':['anpan', 'cream_bread', 'pancake', 'peanut_bread', 'milk_france', 'melonpan'],
'tofu_dish':['sundubu_soup__jun_tofu_pot', 'silken_tofu', 'mabo_tofu', 'firm_tofu', 'cold_tofu'],
'udon':['udon_over', 'kitsune_udon', 'meat_noodles'],
'wagashi':['castella', 'flour_cake', 'dorayaki', 'steamed_bun'],
'pasta':['shrimp_tomato_cream_spaghetti', 'carbonara', 'cream_of_mushroom_spaghetti', 'salmon_cream_pasta', 'cod_roe_spaghetti',
              'tomato_sauce_spaghetti', 'penne_arrabiata', 'meat_sauce_spaghetti'],
'cutlet':['ground_meat_cutlet', 'pork_cutlet', 'pork_cutlet_on_rice', 'chicken_cutlet', 'chicken_mince_cutlet',
               'miso_katsu', 'cheese_katsu'],
'salad':['pumpkin_salad', 'burdock_salad', 'chicken_salad', 'tuna_salad', 'potato_salad', 'macaroni_salad', 'shrimp_avocado_salad', 'kinuasarada',
'cheese_salad', 'seaweed_salad', 'raw_ham_and_salad', 'raw_spring_rolls', 'beans_and_vegetable_salad', 'egg_salad', 'cabbage_shredded',
              'tomato', 'petit_tomatoes', 'avocado_and_raw_vegetable_salad', 'coleslaw_salad', 'caesar_salad', 'steam_vegetables', 'tomato_and_lettuce_salad',
              'mixed_vegetable_salad', 'warm_vegetable_salad', 'radish_salad', 'vegetable_salad'],
'noodle':['forced_udon', 'topped_udon','mabo_vermicelli', 'soup_vermicelli', 'source_chow_mein', 'japchae', 'pho', 'salt_fried_noodles', 'gomoku_yakisoba', 'stir-fried_udon',
               'somen', 'tonkotsu_ramen', 'pork_soy_sauce_ramen', 'salt_ramen', 'soy_source_ramen', 'miso_ramen', 'kakiagesoba', 'kakesoba', 'zarusoba', 'tanukisoba', 'tororo_buckwheat',
               'udon_over', 'kitsune_udon', 'meat_noodles'],
'soup':['miso_soup_with_pork_and_vegetables', 'miso_soup', 'cabbage_soup', 'corn_potage', 'consomme_soup', 'seaweed_soup',
             'ingredients_vegetable_soup', 'vegetable_soup', 'egg_soup'],
'casserole':['udonsuki', 'kimchi_hot_pot', 'sukiyaki', 'chige_nabe', 'weight-gaining_stew_for_sumo',
                  'minced_fish_pot', 'motsunabe', 'yosenabe', 'mizutaki_of_chicken', 'pork_shabu-shabu',
                  'vegetable_pot', 'sundubu_soup__jun_tofu_pot'],
'bowl_of_rice_with_food_on_top': ['unaju', 'unadon', 'shrimp_tendon',
                                       'pork_cutlet_on_rice', 'mixed_rice',
                                       'loco_moco', 'seafood_sushi',
                                       'seafood_don', 'beef_bowl', 'oyakodon',
                                       'chukadon',
                                       'bowl_of_rice_and_fried_fish',
                                       'dolphin_don'],
'aemono': ['namul', 'spinach_shiraae'],
'bean': ['salad_beans', 'green_soybeans', 'natto'],
'bread-table_roll': ['walnut_bread', 'croissant', 'toast', 'butter_roll',
                          'raisin_roll'],
'beef': ['sirloin_steak', 'dice_steak', 'tomato_curry', 'fillet_steak',
              'bulgogi', 'roast_beef', 'yakiniku__karubi'],
'boiled_food': ['boiled_pumpkin', 'roll_cabbage', 'boiled_radish',
                     'boiled_chikuzen', 'meat_and_potatoes'],
'candy-gum': ['candy', 'gum', 'caramel_candy', 'gummy'],
'cereal': ['granola', 'cornflakes', 'brown_rice_flakes'],
"""











































