import React, { Component } from 'react';


/*This is an item  that takes in a list to create the navigation bar*/


class NavBar extends Component {
    render() {
        const { list } = this.props; // Extract the list prop

        return (
            <div>
                <ul>
                    {list.map((item, index) => (
                        <li key={index}>{item}</li>
                    ))}
                </ul>
            </div>
        );
    }
}

export default NavBar;