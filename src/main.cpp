#include "app.hpp"
#include <iostream>



int main(){

    rayos::App App{};
    

    try{
        App.run();
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;

}

