#ifndef AUX_ARRAYS_H
#define AUX_ARRAYS_H

#include <cstddef>
#include <cassert>

namespace arrays {
  template<class T>
  class Array {
  protected:
    T *data;
    bool allocated { false };
  public:
    ~Array() { delete []data; };
  };

  template<class T>
  class OneDArray : public Array<T> {
  private:
    std::size_t n1;
  public:
    OneDArray() {};
    OneDArray(std::size_t n1_);
    void allocate(std::size_t n1_);
    void fillWith(T value);
    void set(std::size_t i1, T value);
    T get(std::size_t i1);
    std::size_t getDim(short int n);
  };
  
  template<class T>
  class TwoDArray : public Array<T> {
  private:
    std::size_t n1, n2;
  public:
    TwoDArray() {};
    TwoDArray(std::size_t n1_, std::size_t n2_);
    void allocate(std::size_t n1_, std::size_t n2_);
    void fillWith(T value);
    void set(std::size_t i1, std::size_t i2, T value);
    T get(std::size_t i1, std::size_t i2);
    std::size_t getDim(short int n);
  };

  template<class T>
  class ThreeDArray : public Array<T> {
  private:
    std::size_t n1, n2, n3;
  public:
    ThreeDArray() {};
    ThreeDArray(std::size_t n1_, std::size_t n2_, std::size_t n3_);
    void allocate(std::size_t n1_, std::size_t n2_, std::size_t n3_);
    void fillWith(T value);
    void set(std::size_t i1, std::size_t i2, std::size_t i3, T value);
    T get(std::size_t i1, std::size_t i2, std::size_t i3);
    std::size_t getDim(short int n);
  };

  // constructor
  template<class T>
  OneDArray<T>::OneDArray(std::size_t n1_) {
    this->allocate(n1_);
  }
  template<class T>
  TwoDArray<T>::TwoDArray(std::size_t n1_, std::size_t n2_) {
    this->allocate(n1_, n2_);
  }
  template<class T>
  ThreeDArray<T>::ThreeDArray(std::size_t n1_, std::size_t n2_, std::size_t n3_) {
    this->allocate(n1_, n2_, n3_);
  }
  
  // allocator
  template<class T>
  void OneDArray<T>::allocate(std::size_t n1_) {
    assert(!(this->allocated) && "# Error: 1d array already allocated.");
    this->data = new T [n1_];
    this->n1 = n1_;
    this->allocated = true;
  } 
  template<class T>
  void TwoDArray<T>::allocate(std::size_t n1_, std::size_t n2_) {
    assert(!(this->allocated) && "# Error: 2d array already allocated.");
    this->data = new T [n2_ * n1_];
    this->n1 = n1_;
    this->n2 = n2_;
    this->allocated = true;
  } 
  template<class T>
  void ThreeDArray<T>::allocate(std::size_t n1_, std::size_t n2_, std::size_t n3_) {
    assert(!(this->allocated) && "# Error: 3d array already allocated.");
    this->data = new T [n3_ * n2_ * n1_];
    this->n1 = n1_;
    this->n2 = n2_;
    this->n3 = n3_;
    this->allocated = true;
  } 

  // filler
  template<class T>
  void OneDArray<T>::fillWith(T value) {
    assert(this->allocated && "# Error: 1d array is not allocated.");
    for (std::size_t n { 0 }; n < this->n1; ++n) {
      this->data[n] = value; 
    }
  }
  template<class T>
  void TwoDArray<T>::fillWith(T value) {
    assert(this->allocated && "# Error: 2d array is not allocated.");
    for (std::size_t n { 0 }; n < this->n2 * this->n1; ++n) {
      this->data[n] = value; 
    }
  }
  template<class T>
  void ThreeDArray<T>::fillWith(T value) {
    assert(this->allocated && "# Error: 3d array is not allocated.");
    for (std::size_t n { 0 }; n < this-> n3 * this->n2 * this->n1; ++n) {
      this->data[n] = value; 
    }
  }

  // element setter
  template<class T>
  void OneDArray<T>::set(std::size_t i1, T value) {
    assert((i1 < this->n1) && "# Error: i1 out of range for 1d array.");
    this->data[i1] = value;
  }
  template<class T>
  void TwoDArray<T>::set(std::size_t i1, std::size_t i2, T value) {
    assert((i1 < this->n1) && "# Error: i1 out of range for 2d array.");
    assert((i2 < this->n2) && "# Error: i2 out of range for 2d array.");
    this->data[i1 + (this->n1) * i2] = value;
  }
  template<class T>
  void ThreeDArray<T>::set(std::size_t i1, std::size_t i2, std::size_t i3, T value) {
    assert((i1 < this->n1) && "# Error: i1 out of range for 3d array.");
    assert((i2 < this->n2) && "# Error: i2 out of range for 3d array.");
    assert((i3 < this->n3) && "# Error: i3 out of range for 3d array.");
    this->data[i1 + (this->n1) * (i2 + (this->n2) * i3)] = value;
  }

  // element getter
  template<class T>
  T OneDArray<T>::get(std::size_t i1) {
    assert((i1 < this->n1) && "# Error: i1 out of range for 1d array.");
    return this->data[i1];
  }
  template<class T>
  T TwoDArray<T>::get(std::size_t i1, std::size_t i2) {
    assert((i1 < this->n1) && "# Error: i1 out of range for 2d array.");
    assert((i2 < this->n2) && "# Error: i2 out of range for 2d array.");
    return this->data[i1 + (this->n1) * i2];
  }
  template<class T>
  T ThreeDArray<T>::get(std::size_t i1, std::size_t i2, std::size_t i3) {
    assert((i1 < this->n1) && "# Error: i1 out of range for 3d array.");
    assert((i2 < this->n2) && "# Error: i2 out of range for 3d array.");
    assert((i3 < this->n3) && "# Error: i3 out of range for 3d array.");
    return this->data[i1 + (this->n1) * (i2 + (this->n2) * i3)];
  }

  // size getter
  template<class T>
  std::size_t OneDArray<T>::getDim(short int n) {
    assert((n == 1) && "# Error: there is only 1 dimension for 1d array.");
    return this->n1;
  }
  template<class T>
  std::size_t TwoDArray<T>::getDim(short int n) {
    assert(((n == 1) || (n == 2)) && "# Error: there are only 2 dimensions for 2d array.");
    if (n == 1) {
      return this->n1;
    } else {
      return this->n2;
    }
  }
  template<class T>
  std::size_t ThreeDArray<T>::getDim(short int n) {
    assert(((n == 1) && (n == 2) && (n == 3)) && "# Error: there are only 3 dimensions for 3d array.");
    if (n == 1) {
      return this->n1;
    } else if (n == 2) {
      return this->n2;
    } else {
      return this->n3;
    }
  }
}

#endif
