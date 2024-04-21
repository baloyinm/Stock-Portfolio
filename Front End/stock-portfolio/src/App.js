import React from 'react';
import logo from './logo.svg';
import './App.css';
import NavBar from './Components/Navbar'


/* All variables  used for function */

const navigation = ['Home','Register', 'Log In', 'About Us', 'Contact Us']
function App() {
  return (
    <div className="App">
      <div className ="NavigationBar">
        <NavBar 
        list={navigation}
        />
      </div>



      <div className ="Body">
      </div>

      <div className ="Footer">
      </div>


    </div>
    
  );
}

export default App;
