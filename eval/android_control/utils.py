import re
import ast
from typing import List, Tuple, Optional, Union
from loguru import logger

# can deal with floating point numbers
class UnifiedCoordinateProcessor:

    def extract_coordinates(self, content: str) -> Tuple[List[float], bool, str]:
        answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        search_content = answer_match.group(1) if answer_match else content
        
        coords = None
        
        coords = self._extract_click_coordinates(search_content)
        if coords:
            coord_type = "4d" if len(coords) == 4 else "2d"
            return coords, True, coord_type
            
        coords = self._extract_bbox_4d(search_content)
        if coords:
            return coords, True, "4d"
            
        coords = self._extract_coord_2d(search_content)
        if coords:
            return coords, True, "2d"
            
        if answer_match:
            coords = self._extract_click_coordinates(content)
            if coords:
                coord_type = "4d" if len(coords) == 4 else "2d"
                return coords, True, coord_type
        
        logger.warning("No coordinates found in the content")
        return None, False, "unknown"

    def modify_coordinates(self, content: str, new_values: List[Union[int, float]]) -> str:
        answer_match = re.search(r'(<answer>)(.*?)(</answer>)', content, re.DOTALL)
import ast
from typing import List, Tuple, Optional, Union
from loguru import logger


class UnifiedCoordinateProcessor:

    def extract_coordinates(self, content: str) -> Tuple[List[float], bool, str]:
        # <|box_start|>(959, 141), (981, 182)<|box_end|>
        box_match = re.search(r'<\|box_start\|>(.*?)<\|box_end\|>', content, re.DOTALL)
        if box_match:
            if box_match.group(1) is not None:
                search_content = box_match.group(1)
                # logger.debug(f"Extracted <|box_start|>(.*?)<|box_end|> content: {search_content}")
                numbers_as_strings = re.findall(r'\d+\.?\d*', search_content)
                if len(numbers_as_strings) == 4:
                    return [float(n) for n in numbers_as_strings], True, "4d"
                elif len(numbers_as_strings) == 2:
                    return [float(n) for n in numbers_as_strings], True, "2d"
                else:
                    logger.warning("No valid coordinates found in <|box_start|> content")
                    return None, False, "unknown"
        answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        if answer_match:
            search_content = answer_match.group(1) if answer_match.group(1) is not None else content
        else:
            search_content = content

        coords = None
        
        coords = self._extract_click_coordinates(search_content)
        if coords:
            coord_type = "4d" if len(coords) == 4 else "2d"
            return coords, True, coord_type
            
        # [x1,y1,x2,y2]
        coords = self._extract_bbox_4d(search_content)
        if coords:
            return coords, True, "4d"
            
        coords = self._extract_coord_2d(search_content)
        if coords:
            return coords, True, "2d"
            
        if answer_match:
            coords = self._extract_click_coordinates(content)
            if coords:
                coord_type = "4d" if len(coords) == 4 else "2d"
                return coords, True, coord_type
        
        logger.warning(f"No coordinates found in the content {content}")
        return None, False, "unknown"

    def modify_coordinates(self, content: str, new_values: List[Union[int, float]]) -> str:
        answer_match = re.search(r'(<answer>)(.*?)(</answer>)', content, re.DOTALL)
        
        if answer_match:
            answer_content = answer_match.group(2)
            modified_answer = self._modify_coordinates_in_text(answer_content, new_values)
            
            if modified_answer != answer_content:
                return (content[:answer_match.start(2)] + 
                       modified_answer + 
                       content[answer_match.end(2):])
        
        return self._modify_coordinates_in_text(content, new_values)

    def _extract_click_coordinates(self, text: str) -> Optional[List[float]]:
        click_matches = re.finditer(r'click\s*\([^)]*\)', text, re.IGNORECASE)
        
        for match in click_matches:
            click_str = match.group(0)
            coords = self._parse_click_content(click_str)
            if coords:
                return coords
        
        return None

    def _parse_click_content(self, click_str: str) -> Optional[List[float]]:
        start_match = re.search(r"start_box=['\"]([^'\"]+)['\"]", click_str)
        end_match = re.search(r"end_box=['\"]([^'\"]+)['\"]", click_str)
        
        if start_match and end_match:
            start_coords = self._parse_coordinate_string(start_match.group(1))
            end_coords = self._parse_coordinate_string(end_match.group(1))
            if start_coords and end_coords and len(start_coords) == 2 and len(end_coords) == 2:
                return start_coords + end_coords
        
        if start_match:
            coords = self._parse_coordinate_string(start_match.group(1))
            if coords:
                return coords
        
        point_match = re.search(r"point=['\"]([^'\"]+)['\"]", click_str)
        if point_match:
            coords = self._parse_coordinate_string(point_match.group(1))
            if coords:
                return coords

        content_match = re.search(r'click\s*\((.*)\)', click_str, re.IGNORECASE)
        if content_match:
            content = content_match.group(1).strip()
            
            coords = self._parse_coordinate_string(content)
            if coords:
                return coords
                
            numbers = re.findall(r'-?\d+\.?\d*', content)
            if len(numbers) in [2, 4]:
                try:
                    return [float(n) for n in numbers]
                except ValueError:
                    pass
        
        return None

    def _parse_coordinate_string(self, coord_str: str) -> Optional[List[float]]:
        """解析各种格式的坐标字符串"""
        if not coord_str:
            return None
            
        coord_str = coord_str.strip()
        
        # <|box_start|>(x,y)<|box_end|>
        match = re.search(r'<\|box_start\|>\s*\(([^)]+)\)\s*<\|box_end\|>', coord_str)
        if match:
            return self._extract_numbers_from_string(match.group(1))
        
        # <point>x y</point> or <point>x,y</point> 
        match = re.search(r'<point>\s*([^<]+)\s*</point>', coord_str)
        if match:
            return self._extract_numbers_from_string(match.group(1))
        
        # [x,y] or [x1,y1,x2,y2]
        match = re.match(r'\[\s*([^\]]+)\s*\]', coord_str)
        if match:
            return self._extract_numbers_from_string(match.group(1))
        
        # {x,y}
        match = re.match(r'\{\s*([^}]+)\s*\}', coord_str)
        if match:
            return self._extract_numbers_from_string(match.group(1))
        
        # (x,y)
        match = re.match(r'\(\s*([^)]+)\s*\)', coord_str)
        if match:
            return self._extract_numbers_from_string(match.group(1))
        
        return self._extract_numbers_from_string(coord_str)

    def _extract_numbers_from_string(self, text: str) -> Optional[List[float]]:
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
        if len(numbers) in [2, 4]:
            try:
                return [float(n) for n in numbers]
            except ValueError:
                pass
        return None

    def _extract_bbox_4d(self, text: str) -> Optional[List[float]]:
        match = re.search(r'\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]', text)
        if match:
            return [float(match.group(i)) for i in range(1, 5)]
        return None

    def _extract_coord_2d(self, text: str) -> Optional[List[float]]:
        match = re.search(r'\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]', text)
        if match:
            return [float(match.group(1)), float(match.group(2))]
        
        match = re.search(r'\{(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\}', text)
        if match:
            return [float(match.group(1)), float(match.group(2))]
        
        match = re.search(r'\{\s*.*\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)\s*.*\}', text)
        if match:
            return [float(match.group(1)), float(match.group(2))]

        match = re.search(r'(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)', text)
        if match:
            return [float(match.group(1)), float(match.group(2))]
        
        return None

    def _modify_4d_coordinates(self, text: str, values: List[Union[int, float]]) -> str:

        pattern = r'\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]'
        replacement = f"[{values[0]}, {values[1]}, {values[2]}, {values[3]}]"
        modified, count = re.subn(pattern, replacement, text, count=1)
        if count > 0:
            return modified
        
        return text

    def _modify_2d_coordinates(self, text: str, values: List[Union[int, float]]) -> str:
        x, y = values[0], values[1]
        
        # <|box_start|>(x,y)<|box_end|> 
        pattern = r'(<\|box_start\|>\()(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)(\)<\|box_end\|>)'
        replacement = f'\\g<1>{x}, {y}\\g<4>'
        modified, count = re.subn(pattern, replacement, text, count=1)
        if count > 0:
            return modified
        
        # <point>x y</point>
        pattern = r'(<point>)(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)(</point>)'
        replacement = f'\\g<1>{x} {y}\\g<4>'
        modified, count = re.subn(pattern, replacement, text, count=1)
        if count > 0:
            return modified
        
        # [x,y]
        pattern = r'\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]'
        replacement = f"[{x}, {y}]"
        modified, count = re.subn(pattern, replacement, text, count=1)
        if count > 0:
            return modified
        
        # {x,y}
        pattern = r'\{(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\}'
        replacement = f"{{{x}, {y}}}"
        modified, count = re.subn(pattern, replacement, text, count=1)
        if count > 0:
            return modified
        
        # {text(x,y)text}
        pattern = r'(\{\s*.*\()(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)(\)\s*.*\})'
        replacement = f'\\g<1>{x}, {y}\\g<4>'
        modified, count = re.subn(pattern, replacement, text, count=1)
        if count > 0:
            return modified
        

        pattern = r'(click\s*\(\s*)(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)(\s*\))'
        replacement = f'\\g<1>{x}, {y}\\g<4>'
        modified, count = re.subn(pattern, replacement, text, count=1)
        if count > 0:
            return modified
        
        return text

    def _modify_coordinates_in_text(self, text: str, new_values: List[Union[int, float]]) -> str:
        modified_text = text
        
        if len(new_values) == 4:
            modified_text = self._modify_4d_coordinates(modified_text, new_values)
        elif len(new_values) == 2:
            modified_text = self._modify_2d_coordinates(modified_text, new_values)
        
        return modified_text

_processor = UnifiedCoordinateProcessor()

def extract_coordinates(content: str, **kwargs) -> Tuple[List[float], bool, str]:
    return _processor.extract_coordinates(content)

def modify_coordinates(content: str, new_values: List[Union[int, float]]) -> str:
    return _processor.modify_coordinates(content, new_values)

def extract_action(response: str) -> Optional[str]:
    patterns = [
        r"'action':\s*'(\w+)'", r"'action':\s*(\w+)",
        r'"action":\s*"(\w+)"', r'"action":\s*(\w+)',
        r"Action:\s*(\w+)\("
    ]
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    search_content = answer_match.group(1) if answer_match else response
    
    for pattern in patterns:
        match = re.search(pattern, search_content)
        if match:
            return match.group(1)
    return None

def extract_all_actions(response: str) -> list[str]:
    pattern = r"""
        'action':\s*'(\w+)' |
        'action':\s*(\w+)   |
        "action":\s*"(\w+)" |
        "action":\s*(\w+)   |
        Action:\s*(\w+)\(
    """
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    search_content = answer_match.group(1) if answer_match else response

    matches = re.findall(pattern, search_content, re.VERBOSE)
    
    # e.g., [('search', '', ''), ('', 'think', '')] -> ['search', 'think']
    all_actions = [item for match_tuple in matches for item in match_tuple if item]
    
    return all_actions

def extract_input_text(content: str) -> str:
    patterns = [r"'input_text':\s*'(.*?)'", r'"input_text":\s*"(.*?)"']
    answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    search_content = answer_match.group(1) if answer_match else content
    
    for pattern in patterns:
        match = re.search(pattern, search_content)
        if match:
            return match.group(1)
    return "no input text"

def extract_coord(content: str) -> Tuple[List[int], bool]:
    coords, success, coord_type = extract_coordinates(content)
    if success and coord_type in ['2d', 'click'] and len(coords) >= 2:
        return [int(coords[0]), int(coords[1])], True
    raise ValueError("No valid 2D coordinates found in the response")

def extract_bbox(response: str) -> Tuple[List[int], bool]:
    coords, success, coord_type = extract_coordinates(response)
    if success and (coord_type == '4d' or len(coords) == 4):
        return [int(c) for c in coords], True
    raise ValueError(f"No valid 4D coordinates found in the {response}")

def extract_click_coordinates(content: str) -> Optional[Tuple[float, ...]]:
    coords, success, _ = extract_coordinates(content)
    return tuple(coords) if success else None

def parse_click_coord(action: str) -> Optional[Tuple[float, ...]]:
    coords, success, _ = extract_coordinates(action)
    return tuple(coords) if success else None

def modify_string_with_new_data(original_content: str, new_numeric_values: List[int]) -> str:
    return modify_coordinates(original_content, new_numeric_values)








if __name__ == "__main__":
    test_cases = [
        "<answer>{'action': 'click', 'coordinate': [877.2, 404, 1477, 672]}</answer>",
        "<answer>[{'action': 'click' ,'coordinate': [447.36, 14.04, 547.2, 36.72] }]</answer>",
        "<answer>{(50, 60) some text}</answer>",
        "click(start_box='<|box_start|>(100,200)<|box_end|>')",
        "click(start_box='<|box_start|> ( 100 , 200 ) <|box_end|>')",
        "click(start_box='[300, 400]')",
        "click(point='<point>250 350</point>')",
        "click(point='<point>250,350</point>')",
        "click(start_box='<|box_start|>(50,60)<|box_end|>', end_box='<|box_start|>(150,160)<|box_end|>')",
        "click(start_box='(1912,1141)')",
        "click(start_box='(87,36)')",
        "click(start_box=\'(90,370)\')",
        "click(point='(1120,421)')",
        "click(point='(13 546)')", 
        "click(66, 77)",
        "click([66,77])",
        "click([66,77,88,99])", 
        "[123, 456]",
        "{789, 012}",
    ]
    
    for test_case in test_cases:
        coords, success, coord_type = extract_coordinates(test_case)
        logger.info(f"'{test_case}' -> coords: {coords}, success: {success}, type: {coord_type}")

    test_modify_cases = [
        ("<answer>click(start_box='<|box_start|>(100,200)<|box_end|>')</answer>", [33, 44]),
        ("<answer>click(point='<point>250 350</point>')</answer>", [55, 66]),
        ("<answer>click(66, 77)</answer>", [11, 22]),  
        ("<answer>[10, 20, 30, 40]</answer>", [100, 200, 300, 400]),
        ("<answer>{'p': (10,20)}</answer>", [1, 2])
    ]
    
    for original, new_vals in test_modify_cases:
        modified = modify_coordinates(original, new_vals)
        print(f"Original: {original}")
        logger.info(f"New: {new_vals} -> Modified: {modified}")
        print()


    action_cases = [
        "<answer>{'action': 'click', 'coordinate': [877, 404, 1477, 672]}</answer>",
        "<answer>{'action': 'type', 'input_text': 'Hello World'}</answer>",
        "Action: click(100, 200)",
        "Action: type('Sample text')",
        "No action here"
    ]
    for case in action_cases:
        action = extract_action(case)
        logger.info(f"'{case}' -> Action: {action}")