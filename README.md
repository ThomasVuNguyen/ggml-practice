# ggml-practice
GGML is awesome and I wanna use it. Thanks

1. Download ggml

git clone https://github.com/ggml-org/ggml
cd ggml

# install python dependencies in a virtual environment
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# build the examples
mkdir build && cd build
cmake ..
cmake --build . --config Release -j 8

2. Test GGML 

From within ggml/build/ folder
# run the GPT-2 small 117M model
../examples/gpt-2/download-ggml-model.sh 117M
./bin/gpt-2-backend -m models/gpt-2-117M/ggml-model.bin -p "This is an example"

3. Try out vector_add.c (4.4.25)

gcc -o vector_add vector_add.c -I./ggml/include ./ggml/build/src/libggml.so ./ggml/build/src/libggml-base.so ./ggml/build/src/libggml-cpu.so -lm

./vector_add

4. Measure time
time ./vector_add

5. Test with math.h

gcc -o vector_add_math vector_add_math.c -lm

6. Performance comparison with more complex matrix ops

gcc -o matrix_math matrix_math.c -lm -O3

gcc -o matrix_ggml matrix_ggml.c ./ggml/build/src/libggml.so ./ggml/build/src/libggml-base.so ./ggml/build/src/libggml-cpu.so -I./ggml/include -lm -O3

Result: math.h 0.37s & ggml.h 0.33s