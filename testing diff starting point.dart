import 'dart:math';
import 'dart:collection';

void main() {

  var map = Map<String, int>();
  
  	// inc learningRate from 0 to 1 
    for (var learningRate = 0.01; learningRate < 1 ; learningRate+=0.01) {
      num x=2;
      num y=7;
      bool found = false;
      
      for (var i = 1; i <= 2000 && (!found); i++) {
        num _deltaX = learningRate * df_dx(x);
        num _deltaY = learningRate * df_dy(y);
        
        x = x - _deltaX;	
        y = y - _deltaY;	

        if(x==2.0 && y==3.0){
          map.putIfAbsent( learningRate.toStringAsPrecision(2) , ()=> i);
          found = true;
        }
      }
    }
  
  var sortedMap = SplayTreeMap<String, dynamic>.from( map, (a, b) => map[a].compareTo(map[b]));
  sortedMap.forEach((k,v)=>{
    print("The learingRate $k took $v  iteration")
  });

}


num f(num x , num y){
  return ((1/3)* pow(x,3)) - 4*x + ((1/3) * pow(y,3)) - 9*y;
} 

num df_dx(num x){
  return  pow(x,2) - 4;
} 

num df_dy(num y){
  return  pow(y,2) - 9;
} 