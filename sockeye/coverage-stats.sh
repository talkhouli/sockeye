# ./sockeye.getcoverage.sh log
tmp=$(mktemp)
cat $1 | grep  ". coverage vector: \[" | cut -f4 -d[ | sed 's/[ 0]\+]//g' > $tmp
sum=`cat $tmp | tr ' ' '\n' | sed 's/]//g' | sort | uniq -c | awk 'BEGIN{sum=0;}{sum=sum+$1*$2;}END{print sum}'`
printf  "Number of target words including sentence end: %s\n" "$sum"
echo "frequency stats:"
cat $tmp | tr ' ' '\n' | sed 's/]//g' | sort | uniq -c
rm $tmp
