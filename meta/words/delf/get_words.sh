#!/bin/sh

wget http://storage.googleapis.com/delf/rparis6k_codebook_65536.tar.gz
mkdir rparis6k_codebook_65536
tar -xvzf rparis6k_codebook_65536.tar.gz -C rparis6k_codebook_65536/
rm rparis6k_codebook_65536.tar.gz

wget http://storage.googleapis.com/delf/roxford5k_codebook_65536.tar.gz
mkdir roxford5k_codebook_65536
tar -xvzf roxford5k_codebook_65536.tar.gz -C roxford5k_codebook_65536/
rm roxford5k_codebook_65536.tar.gz

wget http://storage.googleapis.com/delf/rparis6k_codebook_1024.tar.gz
mkdir rparis6k_codebook_1024
tar -xvzf rparis6k_codebook_1024.tar.gz -C rparis6k_codebook_1024
rm rparis6k_codebook_1024.tar.gz

wget http://storage.googleapis.com/delf/roxford5k_codebook_1024.tar.gz 
mkdir roxford5k_codebook_1024
tar -xvzf roxford5k_codebook_1024.tar.gz -C roxford5k_codebook_1024
rm roxford5k_codebook_1024.tar.gz
