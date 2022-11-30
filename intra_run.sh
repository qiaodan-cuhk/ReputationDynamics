#!/bin/sh
python Top_Down_intra.py --seed 23 --episode 100000 --norm 9 --alpha 0.4 --b 5 &
python Top_Down_intra.py --seed 14 --episode 100000 --norm 9 --alpha 0.4 --b 5 &
python Top_Down_intra.py --seed 25 --episode 100000 --norm 9 --alpha 0.4 --b 5 &
python Top_Down_intra.py --seed 42 --episode 100000 --norm 9 --alpha 0.4 --b 5 &
python Top_Down_intra.py --seed 33 --episode 100000 --norm 9 --alpha 0.4 --b 5 &
python Top_Down_intra.py --seed 452 --episode 100000 --norm 9 --alpha 0.4 --b 5 &
python Top_Down_intra.py --seed 544 --episode 100000 --norm 9 --alpha 0.4 --b 5 &
python Top_Down_intra.py --seed 723 --episode 100000 --norm 9 --alpha 0.4 --b 5 &
python Top_Down_intra.py --seed 293 --episode 100000 --norm 9 --alpha 0.4 --b 5 &
python Top_Down_intra.py --seed 833 --episode 100000 --norm 9 --alpha 0.4 --b 5 
