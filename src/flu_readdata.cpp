#include "flu_readdata.h"

// struct tm string_to_tmstruct( string str, int format_type );
// int daysdiff( struct tm a, struct tm b );

void readdata( std::string filename, std::vector< std::vector<double> > &DATA, int numcol, bool header  )
{
    DATA.clear();
    assert( DATA.size() == 0 );

    std::ifstream infile( filename.c_str(), std::ifstream::in );
    
    //TODO need some error handling here if the file above is not found

    std::string strHeader;
    if( header ) getline(infile, strHeader);

    int col=0;
    int row=0;
    
    int counter=0;
    
    while( true )
    {
        counter++;

        std::string str("");
        infile >> str;
	
        //fprintf( stderr, "\n\t%d \t%d \t%d \t %1.5f", (int)str.length(), row, col, atof( str.c_str() ) ); fflush(stderr);
	
        if( str.length()==0 && infile.eof() ) break;
	
        // quick & dirty check to make sure the string is a float
        //assert( isdigit( str[0] ) || str[0]=='-' || str[0]=='.' );
	
        // if this is the first column, then it's a new row and we need
        // to allocate a new vector
        if( col==0 )
        {
            std::vector<double> vd(numcol);
            DATA.push_back( vd );
        }

        DATA[row][col] = atof( str.c_str() );
	
	
        if( col==numcol-1 ) // if the previous statement just read in the last column
        {
            col=0;
            row++;
        }
        else
        {
            col++;
        }
	
    }
    
    infile.close();
    return;
}




