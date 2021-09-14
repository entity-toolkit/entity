#include "arrays.h"
#include "arrays.cpp"

template class ntt::arrays::Array<int>;
template class ntt::arrays::Array<float>;
template class ntt::arrays::Array<double>;
template class ntt::arrays::Array<bool>;

template class ntt::arrays::OneDArray<int>;
template class ntt::arrays::OneDArray<float>;
template class ntt::arrays::OneDArray<double>;
template class ntt::arrays::OneDArray<bool>;

template class ntt::arrays::TwoDArray<int>;
template class ntt::arrays::TwoDArray<float>;
template class ntt::arrays::TwoDArray<double>;
template class ntt::arrays::TwoDArray<bool>;

template class ntt::arrays::ThreeDArray<int>;
template class ntt::arrays::ThreeDArray<float>;
template class ntt::arrays::ThreeDArray<double>;
template class ntt::arrays::ThreeDArray<bool>;

#include "fields.h"
#include "fields.cpp"

template class ntt::fields::OneDField<int>;
template class ntt::fields::OneDField<float>;
template class ntt::fields::OneDField<double>;
template class ntt::fields::OneDField<bool>;

template class ntt::fields::TwoDField<int>;
template class ntt::fields::TwoDField<float>;
template class ntt::fields::TwoDField<double>;
template class ntt::fields::TwoDField<bool>;

template class ntt::fields::ThreeDField<int>;
template class ntt::fields::ThreeDField<float>;
template class ntt::fields::ThreeDField<double>;
template class ntt::fields::ThreeDField<bool>;
