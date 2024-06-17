#!/bin/bash
pip install gdown
mkdir weights

# seed 1
gdown https://drive.google.com/uc?id=1zMFZ04l4FegnFea2fYWzQxWjh_TKHdkt
mv model.pth.tar-2 weights/maple_seed1.pth

# seed 2
gdown https://drive.google.com/uc?id=1hLk4tqv0BRo3fnt6ncElbqCteGPe59DE
mv model.pth.tar-2 weights/maple_seed2.pth

# seed 3
gdown https://drive.google.com/uc?id=1eEMVG8Tsfc9SrzSiawbfycjMNF7g3WpD
mv model.pth.tar-2 weights/maple_seed3.pth

echo "MaPLe weights downloaded. You should now have a 'weights' folder with 'maple_seed1.pth', 'maple_seed2.pth' and 'maple_seed3.pth'"
