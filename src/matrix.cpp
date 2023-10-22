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
            this->fill(0.);
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

        Matrix operator+ (float value) {
            // add to every element
            Matrix sum(this->width, this->height);
            for(int i = 0; i < this->size; i++) {
                sum.data[i] = this->data[i] + value;
            }
            return sum;
        }
        Matrix operator+ (Matrix& other) {
            // element wise addition
            // check for same shape
            if(!(other.width == this->width && other.height == this->height)) {
                std::string message = "matrix shapes (" + 
                    std::to_string(other.width) + ", " + std::to_string(other.height) + ") and (" + 
                    std::to_string(this->width) + ", " + std::to_string(this->height) + ") incompatible for addition.";
                throw std::runtime_error(message);
            }

            Matrix sum(this->width, this->height);
            for(int i = 0; i < this->size; i++) {
                sum.data[i] = this->data[i] + other.data[i];
            }
            return sum;
        }

    private:
        float* data;
};

int main() {
    Matrix mat(2, 2);
    Matrix sum = mat + 1.;
    Matrix sum2 = mat + sum;
    Matrix sum3 = sum2 + 1;

    mat.print();
    sum.print();
    sum2.print();
    sum3.print();

    Matrix b(1, 2);
    Matrix c = mat + b;

    return 0;
}
