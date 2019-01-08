# ./sockeye.getcoverage.sh log
cat $1 | grep  ". coverage vector: \[" | cut -f4 -d[ | sed 's/[ 0]\+]//g'
