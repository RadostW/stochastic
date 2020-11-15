// Original code avaliable on https://rosettacode.org/wiki/Cholesky_decomposition
// Accessed 2020
// License:  GNU Free Documentation License 1.2
#pragma once
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
 
template <typename scalar_type> class matrix {
public:
    matrix(size_t rows, size_t columns)
        : rows_(rows), columns_(columns), elements_(rows * columns) {}
 
    matrix(size_t rows, size_t columns, scalar_type value)
        : rows_(rows), columns_(columns), elements_(rows * columns, value) {}
 
    matrix(size_t rows, size_t columns,
        const std::initializer_list<std::initializer_list<scalar_type>>& values)
        : rows_(rows), columns_(columns), elements_(rows * columns) {
        assert(values.size() <= rows_);
        size_t i = 0;
        for (const auto& row : values) {
            assert(row.size() <= columns_);
            std::copy(begin(row), end(row), &elements_[i]);
            i += columns_;
        }
    }
 
    size_t rows() const { return rows_; }
    size_t columns() const { return columns_; }
 
    const scalar_type& operator()(size_t row, size_t column) const {
        assert(row < rows_);
        assert(column < columns_);
        return elements_[row * columns_ + column];
    }
    scalar_type& operator()(size_t row, size_t column) {
        assert(row < rows_);
        assert(column < columns_);
        return elements_[row * columns_ + column];
    }
    matrix<scalar_type> operator +(const matrix<scalar_type> rhs) const
    {
        assert(rows_ == rhs.rows_);
        assert(columns_ == rhs.columns_);
        std::vector<scalar_type>sum(elements_.size());
        for(size_t i=0;i<elements_.size();i++)
        {
            sum[i] = elements_[i]+rhs.elements_[i];
        }
        return matrix(rows_,columns_,sum);
    }
    matrix<scalar_type> operator -(const matrix<scalar_type> rhs) const
    {
        assert(rows_ == rhs.rows_);
        assert(columns_ == rhs.columns_);
        std::vector<scalar_type>sum(elements_.size());
        for(size_t i=0;i<elements_.size();i++)
        {
            sum[i] = elements_[i]-rhs.elements_[i];
        }
        return matrix(rows_,columns_,sum);
    }
    matrix<scalar_type> operator *(const matrix<scalar_type> rhs) const
    {
        assert(columns_ == rhs.rows_);
        matrix<scalar_type> product(rows_,rhs.columns_);
        for(size_t i=0;i<rows_;i++)
        {
            for(size_t j=0;j<rhs.columns_;j++)
            {
                for(size_t k=0;k<columns_;k++)
                {
                    product(i,j) += operator()(i,k)*rhs(k,j);
                }
            }
        }
        return product;
    }
    matrix<scalar_type> operator*(const scalar_type rhs) const
    {
        matrix<scalar_type> product(rows_,columns_,elements_);
        for(auto& element : product.elements_)
        {
            element *=rhs;
        }
    }

    //Return transpose
    matrix<scalar_type> T() const
    {
        matrix<scalar_type> transpose(columns_,rows_);
        for(size_t i=0;i<rows_;i++)
        {
            for(size_t j=0;j<columns_;j++)
            {
                transpose(j,i) = operator()(i,j);                
            }
        }
        return transpose;
    }

private:
    size_t rows_;
    size_t columns_;
    std::vector<scalar_type> elements_;
    matrix(size_t rows, size_t columns, std::vector<scalar_type> elements)
        : rows_(rows), columns_(columns), elements_(elements) {}

};
 
template <typename scalar_type>
void print(std::ostream& out, const matrix<scalar_type>& a) {
    size_t rows = a.rows(), columns = a.columns();
    for (size_t row = 0; row < rows; ++row) {
        if(row==0)out << "[";
        out <<"[";
        for (size_t column = 0; column < columns; ++column) {
            if (column > 0)
                out << ", ";
            out << a(row, column);
        }
        if(row==rows-1)out << "]]\n";
        else out << "],\n";
    }
}
