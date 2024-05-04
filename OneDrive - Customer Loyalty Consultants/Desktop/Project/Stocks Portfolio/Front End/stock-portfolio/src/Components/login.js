import React, { Component } from 'react';
import './login.css'


class CreateUserPage extends Component {
    constructor(props) {
        super(props);
        this.state = {
            username: '',
            email: '',
            password: ''
        };
    }

    handleInputChange = (event) => {
        const target = event.target;
        const value = target.value;
        const name = target.name;

        this.setState({
            [name]: value
        });
    }

    handleSubmit = (event) => {
        event.preventDefault();
        // Here you can implement logic to submit the form data, such as sending it to a server
        console.log('Form submitted:', this.state);
        // Optionally, you can reset the form fields after submission
        this.setState({
            username: '',
            email: '',
            password: ''
        });
    }

    render() {
        return (
            <div className="grid-container">
                <div className='user-create-login'>
                <h2>Create User</h2>

                </div>
              
                <form onSubmit={this.handleSubmit} className="create-user-form">
                    <div className="form-group">
                        <label>Username:</label>
                        <input
                            type="text"
                            name="username"
                            value={this.state.username}
                            onChange={this.handleInputChange}
                            required
                        />
                    </div>
                    <div className="form-group">
                        <label>Email:</label>
                        <input
                            type="email"
                            name="email"
                            value={this.state.email}
                            onChange={this.handleInputChange}
                            required
                        />
                    </div>
                    <div className="form-group">
                        <label>Password:</label>
                        <input
                            type="password"
                            name="password"
                            value={this.state.password}
                            onChange={this.handleInputChange}
                            required
                        />
                    </div>
                    <div className="form-group">
                        <button type="submit">Create User</button>
                    </div>
                </form>
            </div>
        );
    }
}

export default CreateUserPage;
