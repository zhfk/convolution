/************************************************************
	this project build for convolution
*************************************************************/
1. open the windows command window and type 
		" aoc --list-boards "
		it should output "de1soc_sharedonly"

2. dir to project root directory and type 
		" aoc device/xxx.cl -o bin/xxx.aocx --board de1soc_sharedonly -v --report " 
		to generate the hardware configuration files
		
3. host execute file :
   build : Project -> Bilde Project 
   		