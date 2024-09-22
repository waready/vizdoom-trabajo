# Crear entorno virtual python
python -m venv new-env 
new-env\Scripts\activate

# Instalar dependencias
pip install numpy
pip install scikit-image
pip install tensorflow
pip install tqdm
pip install vizdoom
pip install torch

# Cacodemon

En este ejercicio, el objetivo es aprovechar el framework de ViZDoom para realizar un experimento básico donde el agente (Doom guy) se enfrenta a un Cacodemon en un escenario. El Cacodemon es colocado aleatoriamente en la sala, y el experimento busca observar el comportamiento del agente mientras aprende a identificar y atacar al enemigo. A lo largo del tiempo, se espera verificar si el agente mejora su capacidad para reaccionar y eliminar al Cacodemon, evaluando así su proceso de aprendizaje en un entorno controlado de combate.
Verificación: Observa el comportamiento del agente mientras aprende a disparar al Cacodemon. Verifica si el agente es capaz de identificar al enemigo y cómo su comportamiento mejora a lo largo del tiempo.

# medical kit

El objetivo de este escenario es enseñar al agente a sobrevivir sin saber qué le hace sobrevivir. El agente solo sabe que la vida es preciosa y que la muerte es mala, por lo que debe aprender qué prolonga su existencia y que su salud está relacionada con ello.
 El mapa es un rectángulo que contiene paredes y un suelo verde y ácido que daña al jugador periódicamente. Al principio hay algunos botiquines distribuidos uniformemente por el mapa. De vez en cuando cae un nuevo botiquín del cielo. Los botiquines curan parte de la salud del jugador; para sobrevivir, el agente debe recogerlos. El episodio termina después de la muerte del jugador o tras un tiempo de espera.

