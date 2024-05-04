import React from 'react';
import logo from './logo.svg';
import './App.css';
import NavBar from './Components/Navbar'
import CreateUserPage  from './Components/login';
import Test from './Components/test';


/* All variables  used for function */

const navigation = ['Home','Register', 'Log-In', 'Calculators', ' AI Tools', 'About Us', 'Contact Us']
function App() {
  return (
    <div className="App">
      <div className ="NavigationBar">
        <NavBar 
        list={navigation}
        />
      </div>

      <div>

        <CreateUserPage/>
        <Test/>
      </div>




      <div className ="Body">
      </div>

      <div className ="Footer">
      </div>


    </div>
    
  );
}

export default App;
