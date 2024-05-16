import pygame
from button import Button
from game_utils import *
from sys import exit
import random


pygame.init()
width, height = 1200, 720
SCREEN = pygame.display.set_mode((width, height))
FONT = pygame.font.SysFont('arialblack', size=60)
FONT1 = pygame.font.SysFont(name='arialblack', size=40)
FONT2 = pygame.font.SysFont(name='arialblack', size=25)

pygame.display.set_caption('Connection')
background_image = pygame.image.load("img/pastel.png")
logo = pygame.image.load('img/graph.png')

quick = {
    'RW Jaccard Sim' : 'random_walk_sim',
    'RW KMeans' : 'random_walk_kmeans'
}

pick = 'RW KMeans'

def main_menu():
        
    TEXT = FONT.render('MENU', True, 'white')
    RECT = TEXT.get_rect(center=(300, 120)) 

    # BUTTONS
    PLAY = Button(None, (300, 320), 'PLAY', FONT1, 'White', 'Cyan')
    OPTIONS = Button(None, (300, 420), 'OPTIONS', FONT1, 'White', 'Cyan')
    QUIT = Button(None, (300, 520), 'QUIT', FONT1, 'White', 'Cyan')

    while True:

        MOUSE_POS = pygame.mouse.get_pos()
        SCREEN.blit(background_image, (0, 0))
        SCREEN.blit(logo, (500, 50))
        SCREEN.blit(TEXT, RECT) 
       
        for button in [PLAY, OPTIONS, QUIT]:
            button.changeColor(MOUSE_POS)
            button.update(SCREEN)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if QUIT.checkForInput(MOUSE_POS):
                    pygame.quit()
                    exit()
                if OPTIONS.checkForInput(MOUSE_POS):
                    SCREEN.fill('black')
                    options()
                if PLAY.checkForInput(MOUSE_POS):
                    SCREEN.fill('black')
                    game()

        pygame.display.update()

def game():
    # GAME STATE
    demo_clusters = [
    MiniCluster('Shades of Red', ['BRICK', 'CHERRY', 'ROSE', 'RUBY']),
    MiniCluster('Little bit of a Beverage', ['DROP', 'SPLASH', 'SPOT', 'SPRINKLE']),
    MiniCluster('Choicest', ['BEST', 'CREAM', 'PICK', 'TOP'] ),
    MiniCluster('__ Bath', ['BIRD', 'BUBBLE', 'MUD', 'SPONGE'])
    ]

    path = f'cluster_data/{quick[pick]}'
    clusters = sampler(path)
    c = Connections(clusters)
    queue = set()
    solved = []

    # if it's just too hard, uncomment:
    # print(clusters)

    # CREATE BUTTONS
    button2 = pygame.image.load('img/button3.png')
    button2 = pygame.transform.scale(button2, (180, 60))
    button3 = pygame.image.load('img/button5.png')
    button3 = pygame.transform.scale(button3, (180, 60))
    bar = pygame.image.load('img/bar.png')
    bar = pygame.transform.scale(bar, (1700, 80))
    hilfe = pygame.image.load('img/hilfe.png')
    hilfe = pygame.transform.scale(hilfe, (60,60))

    
    # HARDCODED BUTTONS
    BACK = Button(button2, (100,680), 'BACK', FONT2, 'Black', 'DarkOrange')
    SUBMIT = Button(button3, (935,620), 'SUBMIT', FONT2, 'Black', 'DarkGreen')
    HILFE = Button(hilfe, (1100, 620), None, FONT2, 'Black', 'White')
    BUTTONS = [BACK, SUBMIT, HILFE]

    # create button table
    pos = [180, 200]
    x, y = pos
    button1 = pygame.image.load('img/button.png')
    button1 = pygame.transform.scale(button1, (300, 60))

    
    for stack in c.table:
        a = []
        for i,word in enumerate(stack):
            a.append(Button(button1, (x + i*280, y), word, FONT2, 'White', 'Cyan'))
        c.GAME_BUTTONS.append(a)
        y += 100
    x, y = pos

    _pos = [0,100]
    _x, _y = _pos

    while True:

        MOUSE_POS = pygame.mouse.get_pos()
        SCREEN.blit(background_image, (0, 0))

        # UPDATE LOOP
        for a in c.GAME_BUTTONS:
            for button in a:
                if button.text_input in queue:
                    button.text = button.font.render(button.text_input, True, button.hovering_color)
                    button.update(SCREEN)
                else:
                    button.changeColor(MOUSE_POS)
                    button.update(SCREEN)
        
        for button in BUTTONS:
                button.changeColor(MOUSE_POS)
                button.update(SCREEN)

        for i, pack in enumerate(solved):
            root, words = pack
            xi, yi = _x, _y
            SCREEN.blit(bar, (0, yi + 120*i))
            TEXT = FONT2.render(root.upper(), True, 'white')
            RECT = TEXT.get_rect(center=(xi + 200, yi+20 + 120*i)) 
            SCREEN.blit(TEXT, RECT)
            for j, word in enumerate(words):
                TEXT = TEXT = FONT2.render(word, True, 'white')
                RECT = TEXT.get_rect(center=(xi + 200 + (j * 280), yi+60 + 120*i)) 
                SCREEN.blit(TEXT, RECT)


        # EVENT LOOP
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                for a in c.GAME_BUTTONS:
                    for button in a:
                        if button.checkForInput(MOUSE_POS) and button.text_input not in queue and len(queue) < 4:
                            queue.add(button.text_input)
                        elif button.checkForInput(MOUSE_POS) and button.text_input in queue:
                            queue.remove(button.text_input)

                if BACK.checkForInput(MOUSE_POS):
                    SCREEN.fill('black')
                    main_menu()


                if SUBMIT.checkForInput(MOUSE_POS) and len(queue) == 4:
                    #check clusters
                    for i, cluster in enumerate(c.clusters):
                        if queue == set(cluster.words):
                            s = c.clusters.pop(i)
                            solved.append((s.root, s.words))
                            c.table = c.init_table()
                            c.empty()
                            pos[1] += 100
                            x, y = pos
                            for stack in c.table:
                                a = []
                                for i,word in enumerate(stack):
                                    a.append(Button(button1, (x + i*280, y), word, FONT2, 'White', 'Cyan'))
                                c.GAME_BUTTONS.append(a)
                                y += 100
                            x,y = pos
                            queue = set()
                            SCREEN.fill('green')
                        else:
                            SCREEN.fill('red')

        # WIN CONDITION
        if not c.clusters:
            SCREEN.fill('black')
            win()
                        
        pygame.display.update()

def options():

    TEXT = FONT2.render('Choose Clustering Algorithm', True, 'white')
    RECT = TEXT.get_rect(center=(300, 120)) 
    button2 = pygame.image.load('img/button3.png')
    button2 = pygame.transform.scale(button2, (180, 60))

    global pick

    while True:

        MOUSE_POS = pygame.mouse.get_pos()
        SCREEN.blit(background_image, (0, 0))
        SCREEN.blit(TEXT, RECT)

        RW_KMEANS = Button(None, (300, 270), 'RW KMeans', FONT1, 'White', 'Cyan')
        RW_SIM = Button(None, (300, 330), 'RW Jaccard Sim', FONT1, 'White', 'Cyan')       
        BACK = Button(button2, (100,680), 'BACK', FONT2, 'Black', 'DarkOrange')

        BUTTONS_PICKS = [RW_KMEANS, RW_SIM]
        BUTTONS = [BACK]

        for button in BUTTONS:
            button.changeColor(MOUSE_POS)
            button.update(SCREEN)

        for button in BUTTONS_PICKS:
            if button.text_input == pick:
                        button.text = button.font.render(button.text_input, True, button.hovering_color)
                        button.update(SCREEN)
            else:
                button.changeColor(MOUSE_POS)
                button.update(SCREEN)


        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if BACK.checkForInput(MOUSE_POS):
                        SCREEN.fill('black')
                        main_menu()
                    for button in BUTTONS_PICKS:
                        if button.checkForInput(MOUSE_POS) and button.text_input != pick:
                            pick = button.text_input

        pygame.display.update()

def win():

    brain = pygame.image.load('img/brain.png')
    brain = pygame.transform.scale(brain, (220,220))
    TEXT = FONT.render('YOU WIN', True, 'white')
    RECT = TEXT.get_rect(center=(560, 120)) 

    while True:


        MOUSE_POS = pygame.mouse.get_pos()
        SCREEN.blit(background_image, (0, 0))
        SCREEN.blit(brain, (450,250))
        SCREEN.blit(TEXT, RECT)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        pygame.display.update()

main_menu()