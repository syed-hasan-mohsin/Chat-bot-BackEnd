import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class HomeService {

  constructor(private http: HttpClient) { }
  
  private apiUrl = 'http://127.0.0.1:5000/api'; // Replace with your API URL

  getData(): Observable<any> {
    return this.http.get<any>(this.apiUrl);
  }

  sendQuestion(question: string) {
    const url = `${this.apiUrl}/ask?question=${question}`;
    return this.http.get<any>(url);
  }
}
