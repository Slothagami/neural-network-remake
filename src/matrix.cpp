#include <iostream>

class Matrix {
    public:
        int width;
        int height;
        int size;

        Matrix(int width, int height) {
            this->width  = width;
            this->height = height;
            this->size   = width * height;

            this->data = new float[this->size];
            this->fill(0);
        }

        Matrix(const Matrix& other) {
            // deep copies an existing matrix
        }

        ~Matrix() {
            // destructor for cleaning memory
            delete[] this->data;
        }

        void fill(float value) {
            for(int i = 0; i < this->size; i++) {
                this->data[i] = value;
            }
        }

        void print() {
            std::cout << "[ ";
            for(int i = 0; i < this->size; i++) {
                if((i-1) % this->width == 1) {
                    std::cout << "[ ";
                }

                std::cout << this->data[i] << " ";

                if((i-1) % this->width == 0) {
                    std::cout << "]\n";
                }
            }
        }

        // Matrix operator+ (float other) {
        //     // add to every element
        // }
        // Matrix operator+ (Matrix other) {
        //     // element wise addition
        // }

    private:
        float* data;
};

int main() {
    Matrix mat(2, 2);
    mat.print();

    return 0;
}
