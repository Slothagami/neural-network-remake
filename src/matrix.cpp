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
            std::cout << "\n";
        }

        Matrix operator* (float value) {
            // element wize multiplication
            Matrix prod(this->width, this->height);
            for(int i = 0; i < this->size; i++) {
                prod.data[i] = this->data[i] * value;
            }
            return prod;
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
            this->check_same_shape(other, "addition");

            Matrix sum(this->width, this->height);
            for(int i = 0; i < this->size; i++) {
                sum.data[i] = this->data[i] + other.data[i];
            }
            return sum;
        }

        Matrix operator- (float value) {
            // add negative of the value
            return *this + (-value);
        }
        Matrix operator- (const Matrix& other) {
            this->check_same_shape(other, "subtraction");

            Matrix sum(this->width, this->height);
            for(int i = 0; i < this->size; i++) {
                sum.data[i] = this->data[i] - other.data[i];
            }
            return sum;
        }

    private:
        float* data;

        bool check_same_shape(const Matrix&other, std::string operation) {
            if(!(other.width == this->width && other.height == this->height)) {
                std::string message = "matrix shapes (" + 
                    std::to_string(other.width) + ", " + std::to_string(other.height) + ") and (" + 
                    std::to_string(this->width) + ", " + std::to_string(this->height) + ") incompatible for " +
                    operation + ".";
                throw std::runtime_error(message);
            }
        }
};

int main() {
    Matrix a(2, 2);
    Matrix b(1, 2);
    Matrix sum = a + b;

    return 0;
}
