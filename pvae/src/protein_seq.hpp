#pragma once

#include <string>
#include <utility>
#include <vector>

#include "common_utils.hpp"

/*
    Aminoacids with chemical properties
    Amino acid		Short	Abbrev.	Avg. mass (Da)	pI		pK1(A) 	pK2(B)
    Aspartic acid	D		Asp		133.10384		2.85	1.99	9.90
    Glutamic acid	E		Glu		147.13074		3.15	2.10	9.47
    Cysteine		C		Cys		121.15404		5.05	1.92	10.70
    Asparagine		N		Asn		132.11904		5.41	2.14	8.72
    Phenylalanine	F		Phe		165.19184		5.49	2.20	9.31
    Threonine		T		Thr		119.12034		5.60	2.09	9.10
    Tyrosine		Y		Tyr		181.19124		5.64	2.20	9.21
    Glutamine		Q		Gln		146.14594		5.65	2.17	9.13
    Serine			S		Ser		105.09344		5.68	2.19	9.21
    Methionine		M		Met		149.20784		5.74	2.13	9.28
    Tryptophan		W		Trp		204.22844		5.89	2.46	9.41
    Valine			V		Val		117.14784		6.00	2.39	9.74
    Alanine			A		Ala		89.09404		6.01	2.35	9.87
    Leucine			L		Leu		131.17464		6.01	2.33	9.74
    Isoleucine		I		Ile		131.17464		6.05	2.32	9.76
    Glycine			G		Gly		75.06714		6.06	2.35	9.78
    Proline			P		Pro		115.13194		6.30	1.95	10.64
    Histidine		H		His		155.15634		7.60	1.80	9.33
    Lysine			K		Lys		146.18934		9.60	2.16	9.06
    Arginine		R		Arg		174.20274		10.76	1.82	8.99

    Selenocysteine	U		Sec		168.053			5.47	1.91	10
    Pyrrolysine		O		Pyl		255.31			?		?		?
*/

const std::vector<char> aminoacids = {
    '_', 'D', 'E', 'C', 'N', 'F', 'T', 'Y', 'Q', 'S', 'M', 'W', 'V', 'A', 'L', 'I', 'G', 'P', 'H', 'K', 'R'
};

template<bool exceptions = true>
constexpr int aa_to_idx(const char aminoacid){
    auto f = std::find(aminoacids.begin(), aminoacids.end(), aminoacid);
    if(f != aminoacids.end()){
        return (f - aminoacids.begin());
    }
    if constexpr(exceptions){
        throw std::runtime_error(std::string("Unknown aminoacid: ") + std::to_string(aminoacid));
    }
    return -1;
}

template<bool exceptions = true>
constexpr char idx_to_aa(const int index){
    if(index < aminoacids.size()){
        return aminoacids[index];
    }
    if constexpr(exceptions){
        throw std::runtime_error(std::string("Unknown aminoacid index: ") + std::to_string(index));
    }
    return -1;
}

std::string randseq(size_t length, size_t max_length){
    if(max_length < length) throw std::runtime_error(std::string("max_length (") + std::to_string(max_length) + ") < length (" + std::to_string(length) +")");
    std::string out(max_length, '_');
    for(size_t i = 0; i < length; ++i){
        out[i] = idx_to_aa(randint(1, 20));
    }
    return out;
}
