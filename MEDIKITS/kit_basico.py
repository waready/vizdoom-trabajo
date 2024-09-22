import requests
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution, GameVariable
import random
import os

# URLs de los archivos en GitHub
config_url = "https://raw.githubusercontent.com/Farama-Foundation/ViZDoom/master/scenarios/health_gathering.cfg"
wad_url = "https://raw.githubusercontent.com/Farama-Foundation/ViZDoom/master/scenarios/health_gathering.wad"

# Función para descargar archivos
def descargar_archivo(url, nombre_archivo):
    response = requests.get(url)
    with open(nombre_archivo, "wb") as file:
        file.write(response.content)
    print(f"{nombre_archivo} descargado correctamente.")

# Descargar los archivos necesarios
descargar_archivo(config_url, "health_gathering.cfg")
descargar_archivo(wad_url, "health_gathering.wad")

# Verificar si los archivos existen antes de continuar
if not os.path.exists("health_gathering.cfg") or not os.path.exists("health_gathering.wad"):
    print("Error: Los archivos necesarios no fueron descargados correctamente.")
    exit(1)

# Configuración del juego
game = DoomGame()
game.load_config("health_gathering.cfg")  # Ruta al archivo descargado
game.set_doom_scenario_path("health_gathering.wad")  # Ruta al archivo descargado

game.set_screen_resolution(ScreenResolution.RES_640X480)
game.set_screen_format(ScreenFormat.RGB24)
game.set_mode(Mode.PLAYER)
game.init()

# Acciones disponibles
actions = [[1, 0, 0],  # Moverse hacia adelante
           [0, 1, 0],  # Girar izquierda
           [0, 0, 1]]  # Girar derecha

# Ejecución de episodios
episodes = 10
for episode in range(episodes):
    game.new_episode()
    while not game.is_episode_finished():
        state = game.get_state()
        game_variables = state.game_variables
        screen_buffer = state.screen_buffer

        # Selección de acción aleatoria
        action = random.choice(actions)
        game.make_action(action)

        print(f"Salud: {game.get_game_variable(GameVariable.HEALTH)}")

    print(f"Resultado del episodio {episode + 1}: {game.get_total_reward()}")

game.close()